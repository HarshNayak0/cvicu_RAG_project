import pathlib
import re


from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


pipeline_options = PdfPipelineOptions(generate_picture_images=True, do_ocr=False, do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = False
converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})

print("Converter Created")

path = "drugs/"

filename = "ampicillin.pdf"

result = converter.convert(f"{path}{filename}")
print(f"Processing {filename}")
filename = re.sub(".pdf", ".md", filename)
md_text = result.document.export_to_markdown()

pathlib.Path(f"clean_text/{filename}").write_bytes(md_text.encode())
print(f"{filename} saved")