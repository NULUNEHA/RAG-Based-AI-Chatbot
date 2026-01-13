import streamlit as st
import os
from utils import load_pdf_text, split_text
from rag_chain import create_retriever, get_qa_chain

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 RAG Chatbot using Groq + LangChain")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask a question based on the document")

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded successfully!")

    if query:
        with st.spinner("📖 Reading and chunking..."):
            raw_text = load_pdf_text(file_path)
            chunks = split_text(raw_text)
            retriever = create_retriever(chunks)
            qa_chain = get_qa_chain(retriever)

        with st.spinner("🤖 Generating answer..."):
            result = qa_chain.invoke({"query": query})
            st.subheader("💬 Answer:")
            st.write(result["result"])