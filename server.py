"""
server.py
---------
Thin HTTP wrapper around query.py's RAG pipeline. Does not modify query.py —
imports and calls its existing answer() function directly.

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /health           -> liveness check
    POST /query             -> {"question": "...", "session": "...", "chunks": 6, "model": "..."}
"""

from pydantic import BaseModel

from fastapi import FastAPI

import query as query_module

app = FastAPI(title="F1 RAG API")


class QueryRequest(BaseModel):
    question: str
    session: str | None = None
    chunks: int = 6
    model: str = "claude-sonnet-4-6"


@app.on_event("startup")
def _startup() -> None:
    query_module._load_dotenv()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def run_query(req: QueryRequest) -> dict:
    response = query_module.answer(
        req.question,
        session_key=req.session,
        n_chunks=req.chunks,
        model=req.model,
    )
    return {"answer": response}
