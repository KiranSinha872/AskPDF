"""
server.py — FastAPI server for Document Q&A (any PDF upload & query) with SHA-256 caching.
Run: uvicorn server:app --port 8000
"""
import os
import json
import hashlib
import pathlib
import uuid
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from ingest import ingest_pdf
from rag_chain import build_chain

CACHE_FILE = os.path.join(config.DB_DIR, "file_hashes.json")


def _load_hash_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_hash_cache(cache: dict) -> None:
    os.makedirs(config.DB_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the local FastEmbed model once at server startup
    config.get_embeddings()
    yield


app = FastAPI(title="Document Q&A", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = pathlib.Path(__file__).parent / "frontend" / "index.html"


class AskRequest(BaseModel):
    question: str
    doc_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse(str(FRONTEND))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Validate file extension and MIME type
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .pdf files with application/pdf content type are accepted.",
        )

    # Read content and enforce MAX_UPLOAD_MB
    content = await file.read()
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of {config.MAX_UPLOAD_MB}MB.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # SHA-256 caching: check if this file was already indexed
    file_hash = hashlib.sha256(content).hexdigest()
    cache = _load_hash_cache()

    if file_hash in cache:
        cached_entry = cache[file_hash]
        cached_doc_id = cached_entry.get("doc_id")
        # Verify collection still exists in Chroma
        try:
            client = chromadb.PersistentClient(path=config.DB_DIR)
            existing = [c.name for c in client.list_collections()]
            if cached_doc_id in existing:
                return {
                    "doc_id": cached_doc_id,
                    "filename": filename,
                    "pages": cached_entry.get("pages", 0),
                    "chunks": cached_entry.get("chunks", 0),
                    "cached": True,
                }
        except Exception:
            pass

    # Save to uploads/
    doc_id = uuid.uuid4().hex[:12]
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}_{filename}")
    with open(save_path, "wb") as f:
        f.write(content)

    # Index document into Chroma collection named doc_id
    try:
        stats = ingest_pdf(save_path, doc_id)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(exc)}")

    # Update hash cache
    cache[file_hash] = {
        "doc_id": doc_id,
        "filename": filename,
        "pages": stats["pages"],
        "chunks": stats["chunks"],
    }
    _save_hash_cache(cache)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "pages": stats["pages"],
        "chunks": stats["chunks"],
        "cached": False,
    }


@app.post("/ask")
def ask(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    doc_id = body.doc_id.strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id must not be empty.")

    # Check if doc_id collection exists
    try:
        client = chromadb.PersistentClient(path=config.DB_DIR)
        collections = [c.name for c in client.list_collections()]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)}")

    if doc_id not in collections:
        raise HTTPException(
            status_code=404,
            detail=f"Document collection '{doc_id}' not found.",
        )

    try:
        chain = build_chain(doc_id)
        result = chain.invoke(question)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
