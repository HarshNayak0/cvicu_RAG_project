import os
import json
import re
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === DIRECTORY SETUP ===
# Define folder paths for PDFs, cleaned markdown, and chunked output
pdf_folder = Path("drugs/")
output_folder = Path("clean_text/")
chunks_folder = Path("chunks/")

# Create output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(chunks_folder, exist_ok=True)
print("Output Directories Established")

# === PDF TO MARKDOWN CONVERTER ===
# Configure the PDF-to-Markdown converter using Docling
pipeline_options = PdfPipelineOptions(
    generate_picture_images=True,  # Generate visual representations (optional)
    do_ocr=False,  # Disable OCR (assumes digital text)
    do_table_structure=True,  # Keep table structure intact
)
pipeline_options.table_structure_options.do_cell_matching = False

# Initialize the Docling converter
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
print("Converter Created")

# === TEXT SPLITTER ===
# Initialize RecursiveCharacterTextSplitter for semantic chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Max tokens per chunk
    chunk_overlap=100,  # Overlap for contextual coherence
    separators=["\n\n", "\n", ".", " ", ""],  # Prioritize natural breakpoints
)
print("Text Splitter Initialized")


# === FUNCTION: Extract text from PDFs into Markdown format ===
def extract_text_to_md(pdf_folder):
    for pdf_path in pdf_folder.glob("*.pdf"):
        print(f"Processing {pdf_path.name}")

        md_path = output_folder / f"{pdf_path.stem}.md"

        # Convert PDF to markdown using Docling
        result = converter.convert(str(pdf_path))
        md_text = result.document.export_to_markdown()

        # Save markdown to file
        Path(f"{md_path}").write_bytes(md_text.encode())
        print(f"{pdf_path.stem}.md Saved.")


# === FUNCTION: Clean Markdown content using regex ===
def clean_md(output_folder):
    for filename in output_folder.glob("*.md"):
        print(f"Cleaning {filename}...")

        md_text = filename.read_text(encoding="utf-8")

        # Remove common visual artifacts and formatting issues
        md_text = re.sub(r"", "", md_text)  # Bullet points
        md_text = re.sub(r"\_{-}", "", md_text)  # LaTeX artifact
        md_text = re.sub(r"\\?_", "", md_text)  # Underscores
        md_text = re.sub(r"\\$", "", md_text)  # Dollar signs
        md_text = re.split(r"\n#+\s*References\b", md_text, flags=re.IGNORECASE)[
            0
        ]  # Cut off reference section

        # Overwrite file with cleaned content
        filename.write_text(md_text, encoding="utf-8")
        print(f"Cleaned Markdown saved: {filename}")


# === FUNCTION: Find closest preceding headers for a chunk ===
def extract_nearest_headers(chunk_start_idx, full_text):
    lines = full_text[
        :chunk_start_idx
    ].splitlines()  # Only consider text before chunk start
    headers = {"Header 1": "", "Header 2": "", "Header 3": ""}
    for line in reversed(lines):  # Iterate backwards to find nearest headers
        if line.startswith("###") and not headers["Header 3"]:
            headers["Header 3"] = line.lstrip("# ")
        elif line.startswith("##") and not headers["Header 2"]:
            headers["Header 2"] = line.lstrip("# ")
        elif line.startswith("#") and not headers["Header 1"]:
            headers["Header 1"] = line.lstrip("# ")
        if all(headers.values()):
            break  # Stop if all levels found
    return headers


# === FUNCTION: Split Markdown into chunks and attach header metadata ===
def chunk_markdown_text(output_folder, chunks_folder):
    for md_file in output_folder.glob("*.md"):
        print(f"Chunking {md_file.name}...")

        md_text = md_file.read_text(encoding="utf-8")
        chunks = splitter.create_documents([md_text])  # Chunk entire document

        file_chunks = []
        for i, chunk in enumerate(chunks):
            start_idx = md_text.find(
                chunk.page_content.strip()
            )  # Locate chunk position in original text
            header_meta = extract_nearest_headers(
                start_idx, md_text
            )  # Get header metadata

            # Construct chunk object
            file_chunks.append(
                {
                    "file": md_file.name,
                    "chunk_id": f"{md_file.stem}_{i}",
                    "content": chunk.page_content,
                    "metadata": header_meta,
                }
            )

        print(f"{len(file_chunks)} chunks created for {md_file.name}")

        # Save chunked JSON data to disk
        chunk_file = chunks_folder / f"{md_file.stem}_chunks.json"
        chunk_file.write_text(json.dumps(file_chunks, indent=4), encoding="utf-8")
        print(f"Chunk data for {md_file.name} saved")


# === FUNCTION: Generate and store embeddings + build FAISS index ===
def embed_chunks(chunks_folder, model_name="all-MiniLM-L6-v2"):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    all_embeddings = []
    all_metadatas = []

    for chunk_file in chunks_folder.glob("*chunks.json"):
        print(f"Embedding: {chunk_file.name}")
        chunk_data = json.loads(chunk_file.read_text(encoding="utf-8"))

        for chunk in chunk_data:
            # Combine header metadata with content for richer context
            headers = " ".join(chunk.get("metadata", {}).values())
            text = headers + "\n" + chunk["content"]
            embedding = model.encode(text)

            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm != 0:
                embedding = embedding / norm

            all_embeddings.append(embedding)
            all_metadatas.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "file": chunk["file"],
                    "metadata": chunk.get("metadata", {}),
                    "preview": text[:200],  # First 200 characters as snippet
                }
            )

        print(f"Chunks embedded for {chunk_file.name}")

    print(f"Total chunks embedded: {len(all_embeddings)}")

    # Create FAISS index with cosine similarity
    dimension = len(all_embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(all_embeddings).astype("float32"))

    # Save index and metadata to disk for retrieval
    Path("rag_index.faiss").write_bytes(faiss.serialize_index(index))
    Path("rag_metadata.pkl").write_bytes(pickle.dumps(all_metadatas))

    print("FAISS index saved to 'rag_index.faiss'")
    print("Metadata saved to 'rag_metadata.pkl'")


# === DEBUG TOOL: View sample metadata ===
def inspect_metadata(metadata_folder="rag_metadata.pkl"):
    with open(metadata_folder, "rb") as f:
        metadata = pickle.load(f)
        print(metadata[20:22])  # Print sample entries


# === EXECUTION PIPELINE ===
# extract_text_to_md(pdf_folder)
# clean_md(output_folder)
# chunk_markdown_text(output_folder, chunks_folder)
# embed_chunks(chunks_folder)

# Uncomment to inspect metadata manually
# inspect_metadata()
