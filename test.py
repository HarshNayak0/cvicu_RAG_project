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
from langchain.text_splitter import MarkdownHeaderTextSplitter


# Creates Path objects for file directories -> easier to manipulate and cleaner than manually opening files using os
pdf_folder = Path("drugs_final/")
output_folder = Path("clean_text/")
chunks_folder = Path("chunks/")

# Ensures that output folders exist, if not it creates them
os.makedirs(output_folder, exist_ok=True)
os.makedirs(chunks_folder, exist_ok=True)
print("Output Directories Established")

# Initializes PDF to Markdown converter courtesy of Docling -> OCR turned off to prevent text from images from being rendered in md incorrectly, table structure optimized
pipeline_options = PdfPipelineOptions(
    generate_picture_images=True, do_ocr=False, do_table_structure=True
)
pipeline_options.table_structure_options.do_cell_matching = False
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print("Converter Created")

# Parameters for chunking function
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Chunker initialized
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

print("Text Splitter Initialized")


# PDFs to md - written to appropriate output folder
def extract_text_to_md(pdf_folder):
    for pdf_path in pdf_folder.glob("*.pdf"):
        print(f"Processing {pdf_path.name}")

        md_path = output_folder / f"{pdf_path.stem}.md"

        result = converter.convert(str(pdf_path))
        md_text = result.document.export_to_markdown()

        Path(f"{md_path}").write_bytes(md_text.encode())


# md cleaned using regex -> find a more dynamic way of cleaning in later version, regex is poverty
def clean_md(output_folder):

    for filename in output_folder.glob("*.md"):

        print(f"Cleaning {filename}...")

        md_text = filename.read_text(encoding="utf-8")

        # Fix encoding issues & remove unwanted characters
        md_text = re.sub(r"", "", md_text)  # Remove box bullets
        md_text = re.sub(r"\_{-}", "", md_text)  # Remove LaTeX artifacts
        md_text = re.sub(r"\\?_", "", md_text)  # Remove underscores
        md_text = re.sub(r"\$", "", md_text)  # Remove $
        md_text = re.split(r"\n#+\s*References\b", md_text, flags=re.IGNORECASE)[0]

        # Save cleaned Markdown file using pathlib
        filename.write_text(md_text, encoding="utf-8")
        print(f"Cleaned Markdown saved: {filename}")


# md chunked according to headers (best for data set as context is separated by headers)
def chunk_markdown_text(output_folder, chunks_folder):

    for md_file in output_folder.glob("*.md"):

        print(f"Chunking {md_file.name}...")

        md_text = md_file.read_text(encoding="utf-8")

        chunks = markdown_splitter.split_text(md_text)

        file_chunks = [
            {
                "file": md_file.name,
                "chunk_id": f"{md_file.stem}_{i}",
                "content": (
                    chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                ),
                "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
            }
            for i, chunk in enumerate(chunks)
        ]

        print(f"{len(file_chunks)} chunks created for {md_file.name}")

        chunk_file = chunks_folder / f"{md_file.stem}_chunks.json"
        chunk_file.write_text(json.dumps(file_chunks, indent=4), encoding="utf-8")

        print(f"Chunk data for {md_file.name} saved")


def embed_chunks(chunks_folder, model_name="all-MiniLM-L6-v2"):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    all_embeddings = []
    all_metadatas = []

    for chunk_file in chunks_folder.glob("*chunks.json"):
        print(f"Embedding: {chunk_file.name}")
        chunk_data = json.loads(chunk_file.read_text(encoding="utf-8"))

        for chunk in chunk_data:
            text = chunk["content"]
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
                    "preview": text[:200],
                }
            )

        print(f"Chunks embedded for {chunk_file.name}")

    print(f"Total chunks embedded: {len(all_embeddings)}")

    # Build and save FAISS index
    dimension = len(all_embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(all_embeddings).astype("float32"))

    # Save index and metadata
    Path("rag_index.faiss").write_bytes(faiss.serialize_index(index))
    Path("rag_metadata.pkl").write_bytes(pickle.dumps(all_metadatas))

    print("FAISS index saved to 'rag_index.faiss'")
    print("Metadata saved to 'rag_metadata.pkl'")


def inspect_metadata(metadata_folder="rag_metadata.pkl"):

    with open(metadata_folder, "rb") as f:
        metadata = pickle.load(f)

        # Manipulate indexing to retrieve specific chunks
        print(metadata[20:22])


# Pipeline
# extract_text_to_md(pdf_folder)
# clean_md(output_folder)
# chunk_markdown_text(output_folder, chunks_folder)
embed_chunks(chunks_folder)
