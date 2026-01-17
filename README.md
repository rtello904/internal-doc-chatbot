# Internal Document Chatbot

A private chatbot that lets you upload your own documents (like PDFs) and ask questions about them using embeddings, retrieval, and an LLM.  
Documents are processed per session and are not shared globally.

---

## Features

- Upload a document and chat with it  
- Retrieval-augmented generation (RAG)  
- Works with hosted or local LLMs  
- No global document sharing  
- Clean, modular project structure  

---

## Project Structure

internal-doc-chatbot/
```text

├── app.py
├── src/
│ └── embed.py
├── utils/
│ └── loader.py
├── .env
├── .env.example
├── requirements.txt
└── README.md
```

### app.py

Main Streamlit application.

Responsibilities:
- UI (file upload, question input, answer display)
- Load uploaded documents
- Split into chunks
- Generate embeddings
- Build vector store
- Run retrieval + LLM to answer questions

Run with:

```
streamlit run app.py
```

### src/embed.py
Standalone script for pre-embedding documents.

Responsibilities:
- Load documents from disk
- Split into chunks
- Generate embeddings
- Build and optionally save a vector index
- Useful when you want:
- Persistent indexes
- Faster startup
- No re-embedding on every run

Run with:
```bash
python src/embed.py
```

### utils/loader.py

Document loading utilities.

Responsibilities:
- Read PDFs or uploaded files
- Convert files into LangChain Document objects
- Handle temporary file storage
- Keeps file I/O separate from app logic.

### .env

Private environment variables (not committed):

HUGGINGFACEHUB_API_TOKEN=...  
OPENAI_API_KEY=...  

### .env.example

Template for required environment variables:

HUGGINGFACEHUB_API_TOKEN=your-token-here  
OPENAI_API_KEY=your-key-here  


Copy this file to .env and fill in real values.

### requirements.txt

All Python dependencies needed to run the project.

Install with:
```
pip install -r requirements.txt
```
## Setup
1. Clone the repo
```
git clone <your-repo-url>
cd internal-doc-chatbot
```
2. Create environment

Using venv:

```
python -m venv venv
source venv/bin/activate
```

Or conda:
```
conda create -n chatbot python=3.10
conda activate chatbot
```
3. Install dependencies
```pip install -r requirements.txt```

4. Create .env
```cp .env.example .env```


Edit .env and add your API keys.

Run the App
```streamlit run app.py```


Then:
- Upload a PDF
- Ask questions about it
- Get answers from your document

How It Works:
- User uploads a document
- Document is parsed into text
- Text is split into chunks
- Chunks are converted to embeddings
- Embeddings go into a vector store
- User asks a question
- Most relevant chunks are retrieved
- LLM answers using retrieved context

Notes:
- Documents are processed per session
- Nothing is globally shared
- You can switch LLM backends (OpenAI, Hugging Face, or local models)
- embed.py is optional for precomputing embeddings