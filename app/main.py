from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from app.ingest import ingest_pdf
from app.retriever import query_documents, list_sources
from app.chain import answer_question

app = FastAPI(
    title="RAG Research Assistant",
    description="Upload papers, ask questions, get answers with citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    text: str
    source: str
    page: Optional[int] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDFs supported right now")

    contents = await file.read()
    num_chunks = ingest_pdf(contents, file.filename)

    return {
        "filename": file.filename,
        "chunks_indexed": num_chunks,
    }


@app.post("/query", response_model=AnswerResponse)
def query(req: QueryRequest):
    retrieved = query_documents(req.question, top_k=req.top_k)

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="no documents indexed yet, upload a paper first",
        )

    answer, sources = answer_question(req.question, retrieved)

    return AnswerResponse(
        answer=answer,
        sources=[
            SourceChunk(
                text=s["text"],
                source=s["source"],
                page=s.get("page"),
            )
            for s in sources
        ],
    )


@app.get("/documents")
def get_documents():
    return {"documents": list_sources()}
