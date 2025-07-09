from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from utils.loader import load_pdf
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
st.title("📄 Document Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    docs = load_pdf(uploaded_file)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    chunks = chunks[:20]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"temperature": 0.5}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Ask a question about your document:")
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(query)
            st.markdown("### 🧠 Answer")
            st.write(result)
