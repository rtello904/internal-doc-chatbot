import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document

def load_pdf(uploaded_file) -> list[Document]:
    """
    Takes a Streamlit-uploaded file and returns a list of LangChain Document objects.
    This allows you to pass them to a vector store like FAISS.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Use LangChain's PyMuPDFLoader to parse the PDF
    loader = PyMuPDFLoader(tmp_path)
    return loader.load()