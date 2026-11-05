"""FastAPI backend. Run: uvicorn api:app --port 8000"""
import logging
import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent
from config import ARTIFACT_DIR

log = logging.getLogger("api")
app = FastAPI(title="Agentic Knowledge Retrieval API")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


class Source(BaseModel):
    n: int
    source: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: list[Source]
    artifacts: list[str]
    trace: list[str]
    seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return agent.ask(req.question.strip(), req.session_id)
    except Exception:
        log.exception("agent failed")
        raise HTTPException(status_code=500, detail="The agent failed to answer. Check the server logs.")


@app.get("/artifacts/{name}")
def artifact(name: str):
    safe = Path(name).name  # blocks path traversal like ../../secret
    path = ARTIFACT_DIR / safe
    if safe != name or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type=mimetypes.guess_type(safe)[0] or "application/octet-stream")
