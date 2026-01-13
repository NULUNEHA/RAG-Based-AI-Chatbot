from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

def split_text(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
    return splitter.split_text(text)