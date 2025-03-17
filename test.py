import os
import json
import ftfy
import re

from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_folder = Path("drugs/")
output_folder = Path("clean_text/")
chunks_folder = Path("chunks/")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(chunks_folder, exist_ok=True)
print("Output Directories Established")

pipeline_options = PdfPipelineOptions(
    generate_picture_images=True, do_ocr=False, do_table_structure=True
)
pipeline_options.table_structure_options.do_cell_matching = False
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print("Converter Created")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " "]
)

print("Text Splitter Initialized")


def extract_text_to_md(pdf_folder):
    for pdf_path in pdf_folder.glob("*.pdf"):
        print(f"Processing {pdf_path.name}")

        md_path = output_folder / f"{pdf_path.stem}.md"

        result = converter.convert(str(pdf_path))
        md_text = result.document.export_to_markdown()

        Path(f"{md_path}").write_bytes(md_text.encode())


def clean_md(output_folder):

    for filename in output_folder.glob("*.md"):

        print(f"Cleaning {filename}...")

        md_text = filename.read_text(encoding="utf-8")

        # Fix encoding issues & remove unwanted characters
        md_text = re.sub(r"", "", md_text)  # Remove box bullets
        md_text = re.sub(r"\_{-}", "", md_text)  # Remove LaTeX artifacts
        md_text = re.sub(r"\\?_", "", md_text)  # Remove underscores
        md_text = re.sub(r"\$", "", md_text) #Remove $

        # Save cleaned Markdown file using pathlib
        filename.write_text(md_text, encoding="utf-8")
        print(f"Cleaned Markdown saved: {filename}")



def chunk_markdown_text(output_folder, chunks_folder):

    for md_file in output_folder.glob("*.md"):

        print(f"Chunking {md_file.name}...")

        md_text = md_file.read_text(encoding="utf-8")

        chunks = text_splitter.split_text(md_text)

        file_chunks = [
            {
                "file": md_file.name,  
                "chunk_id": f"{md_file.stem}_{i}",  
                "content": chunk,
            }
            for i, chunk in enumerate(chunks)
        ]

        print("Chunks created for {md_file.name}")


        chunk_file = chunks_folder / f"{md_file.stem}_chunks.json"
        chunk_file.write_text(json.dumps(file_chunks, indent=4), encoding="utf-8")

        print(f"Chunk data for {md_file.name} saved")


extract_text_to_md(pdf_folder)
clean_md(output_folder)
chunk_markdown_text(output_folder, chunks_folder)
