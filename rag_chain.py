import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # or use langchain_huggingface
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


load_dotenv()

def create_retriever(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(chunks, embedding=embeddings)
    return vectordb.as_retriever()

def get_qa_chain(retriever):
    llm = ChatGroq(
        model_name="llama3-8b-8192",  # or llama3-70b-8192
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )