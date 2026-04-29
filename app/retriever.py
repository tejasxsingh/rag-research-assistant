import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
COLLECTION_NAME = "research_papers"

# persistent local store so documents survive restarts
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DIR,
    anonymized_telemetry=False,
))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def add_documents(docs: List[Dict]) -> None:
    """embed and store a list of document chunks"""
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    vectors = embeddings.embed_documents(texts)

    # chromadb needs string ids
    existing_count = collection.count()
    ids = [f"doc_{existing_count + i}" for i in range(len(docs))]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )


def query_documents(question: str, top_k: int = 5) -> List[Dict]:
    """retrieve the most relevant chunks for a question"""
    if collection.count() == 0:
        return []

    query_vector = embeddings.embed_query(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "page": results["metadatas"][0][i].get("page"),
            "distance": results["distances"][0][i],
        })

    return retrieved


def list_sources() -> List[str]:
    """return unique document names in the store"""
    if collection.count() == 0:
        return []

    all_docs = collection.get(include=["metadatas"])
    sources = set()
    for meta in all_docs["metadatas"]:
        sources.add(meta.get("source", "unknown"))
    return sorted(sources)
