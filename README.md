# RAG Research Assistant

API that lets you upload research papers and ask questions about them. Uses retrieval-augmented generation to ground answers in the actual paper content and return citations.

## How it works

1. Upload a PDF through the `/upload` endpoint
2. The paper gets split into chunks, embedded with OpenAI's `text-embedding-3-small`, and stored in ChromaDB
3. When you ask a question via `/query`, the most relevant chunks are retrieved by cosine similarity
4. Those chunks are passed as context to GPT-4o-mini, which generates an answer grounded in the paper
5. The response includes the answer plus the source chunks it drew from (with page numbers)

## Tech stack

- **LangChain** for document loading, text splitting, and LLM orchestration
- **ChromaDB** as the vector store (persistent local storage)
- **OpenAI** for embeddings (text-embedding-3-small) and generation (GPT-4o-mini)
- **FastAPI** for the API layer
- **Docker** for containerized deployment
- **pytest** for testing

## Running locally

```bash
# copy env template and add your OpenAI key
cp .env.example .env

# with docker
docker compose up --build

# or without docker
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs at `http://localhost:8000/docs`

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF to index |
| POST | `/query` | Ask a question about uploaded papers |
| GET | `/documents` | List all indexed papers |
| GET | `/health` | Health check |

### Example query

```bash
# upload a paper
curl -X POST http://localhost:8000/upload \
  -F "file=@my_paper.pdf"

# ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology did the authors use?", "top_k": 5}'
```

### Example response

```json
{
  "answer": "The authors used a sparse partial least squares approach...",
  "sources": [
    {
      "text": "We propose a sparse multi-block PLS method that...",
      "source": "my_paper.pdf",
      "page": 4
    }
  ]
}
```

## Running tests

```bash
pytest tests/ -v
```

Tests use mocked LLM and embedding calls so they run without an API key.

## Project structure

```
rag-research-assistant/
  app/
    main.py        # FastAPI endpoints
    ingest.py      # PDF loading and chunking
    retriever.py   # ChromaDB vector store operations
    chain.py       # LLM prompt and answer generation
  tests/
    test_api.py    # API integration tests
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
```
