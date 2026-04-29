import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_rejects_non_pdf():
    resp = client.post(
        "/upload",
        files={"file": ("notes.txt", b"some text", "text/plain")},
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"] or "pdf" in resp.json()["detail"]


def test_query_with_no_documents():
    with patch("app.main.query_documents", return_value=[]):
        resp = client.post("/query", json={"question": "what is PLS?"})
        assert resp.status_code == 404


def test_query_with_documents():
    fake_retrieved = [
        {
            "text": "PLS is a regression technique for high-dimensional data.",
            "source": "test_paper.pdf",
            "page": 3,
            "distance": 0.15,
        }
    ]
    fake_answer = "PLS (Partial Least Squares) is a regression technique commonly used with high-dimensional data."
    fake_sources = [
        {"text": "PLS is a regression technique for high-dimensional data.", "source": "test_paper.pdf", "page": 3}
    ]

    with patch("app.main.query_documents", return_value=fake_retrieved), \
         patch("app.main.answer_question", return_value=(fake_answer, fake_sources)):

        resp = client.post("/query", json={"question": "what is PLS?"})
        assert resp.status_code == 200

        body = resp.json()
        assert "PLS" in body["answer"]
        assert len(body["sources"]) == 1
        assert body["sources"][0]["source"] == "test_paper.pdf"


def test_documents_endpoint():
    with patch("app.main.list_sources", return_value=["paper_a.pdf", "paper_b.pdf"]):
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert len(resp.json()["documents"]) == 2
