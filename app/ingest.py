import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.retriever import add_documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def ingest_pdf(pdf_bytes: bytes, filename: str) -> int:
    """
    split a pdf into chunks and add them to the vector store.
    returns the number of chunks created.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        docs = []
        for chunk in chunks:
            docs.append({
                "text": chunk.page_content,
                "metadata": {
                    "source": filename,
                    "page": chunk.metadata.get("page", 0),
                },
            })

        add_documents(docs)
        return len(docs)

    finally:
        os.unlink(tmp_path)
