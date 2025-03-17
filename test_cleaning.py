import os
import ftfy
import re

file = "clean_text/acetaminophen.md"


def clean_markdown(file):
    if file.endswith(".md"):
        print(f"Cleaning {file}")
        with open(file, "r") as f:
            md_text = f.read()
            md_text = ftfy.fix_text(md_text)
            md_text = re.sub(r"ï[\'’`\"]·", "", md_text)
            md_text = re.sub(r"\$", "", md_text)
            print(md_text)

clean_markdown(file)