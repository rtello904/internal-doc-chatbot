# Ask Your PDF 📄🤖

A Streamlit app that lets you upload PDFs and chat with their content using local Hugging Face models, vector search, and retrieval-augmented generation (RAG).

You upload one or more PDFs → the app chunks the text → embeds it with Sentence-Transformers → stores it in Chroma → and lets you ask natural-language questions about your files using a Flan-T5 language model.

---

## Features

- Upload multiple PDFs
- Automatic text extraction and chunking
- Semantic search using embeddings
- Local vector store with Chroma
- Conversational memory
- Fully local LLM (no OpenAI API required)
- Simple Streamlit UI

---

## Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Sentence-Transformers
- ChromaDB
- PyPDF2

---

## Project Structure

```text
.
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
