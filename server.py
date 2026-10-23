"""
server.py — FastAPI server for the Gas Turbine Maintenance Assistant RAG system.
Run: uvicorn server:app --port 8000
"""
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_chain import build_chain

# Chain is built once at startup
_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain
    _chain = build_chain()
    yield


app = FastAPI(title="Gas Turbine Maintenance Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = pathlib.Path(__file__).parent / "frontend" / "index.html"


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse(str(FRONTEND))


@app.post("/ask")
def ask(body: QuestionRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        result = _chain.invoke(question)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
