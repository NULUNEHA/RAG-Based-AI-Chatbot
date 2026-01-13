# test_loader.py
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("C:\Haswin info\PROJECTS\Mini projects\mini_project.pdf")  # Replace with any PDF path
docs = loader.load()
print(f"Loaded {len(docs)} pages")