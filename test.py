import pathlib
import os
import json


from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_folder_path = "drugs/"
output_folder = "clean_text/"

pipeline_options = PdfPipelineOptions(generate_picture_images=True, do_ocr=False, do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = False
converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})

print("Converter Created")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "]
)

print("Text Splitter Initialized")

os.makedirs(output_folder, exist_ok=True)
print("Output Directory Established")

def extract_text_to_markdown(pdf_folder_path):
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            print(f"Beginning to Process {filename}")
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_path = os.path.join(output_folder, md_filename)

            result = converter.convert(f"{pdf_folder_path}{filename}")
            print(f"Processing {filename}")
            md_text = result.document.export_to_markdown()

            # change back to md_path when done testing
            pathlib.Path("test_clean_text/acetaminophen.md").write_bytes(md_text.encode())
            print(f"{filename} saved")

all_chunks = []
md_path = "clean_text/acetaminophen.md"

def chunk_markdown_text(md_path, all_chunks):

    with open(md_path, "r") as f:
        md_text = f.read()
        print("Accessing acetaminophen.md")

    chunks = text_splitter.split_text(md_text)

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "file": "acetaminophen.md",
            "chunk_id": f"acetaminophen.md_{i}",
            "content": chunk
        })

    print("Chunks created")

    with open("chunks/acetaminophen_chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=4)

    print(f"Chunk data for {md_path} saved")

extract_text_to_markdown(pdf_folder_path)
chunk_markdown_text(md_path, all_chunks)

