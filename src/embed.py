import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils.loader import load_docs
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Path to documents
DATA_DIR = "data"
INDEX_DIR = "embeddings/faiss_index"

def main():
    print("🔍 Loading documents...")
    documents = load_docs(DATA_DIR)

    print(f"📄 Loaded {len(documents)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    print(f"🧠 Creating embeddings for {len(chunks)} chunks...")
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embedding_model)

    print(f"💾 Saving FAISS index to {INDEX_DIR}...")
    vector_store.save_local(INDEX_DIR)
    print("✅ Done. Embeddings ready!")

if __name__ == "__main__":
    main()
