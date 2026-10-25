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
import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from ingest import ingest_pdf
from rag_chain import build_chain

CACHE_FILE = os.path.join(config.DB_DIR, "file_hashes.json")

# In-memory status tracker for document indexing jobs
# doc_id -> {status, batches_done, batches_total, pages, chunks, filename, error}
indexing_jobs = {}


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


def _background_indexer(save_path: str, doc_id: str, file_hash: str, filename: str):
    def progress_callback(batches_done, batches_total, pages, chunks):
        if doc_id in indexing_jobs:
            indexing_jobs[doc_id]["batches_done"] = batches_done
            indexing_jobs[doc_id]["batches_total"] = batches_total
            indexing_jobs[doc_id]["pages"] = pages
            indexing_jobs[doc_id]["chunks"] = chunks

    try:
        stats = ingest_pdf(save_path, doc_id, progress_callback=progress_callback)
        batches_total = (stats["chunks"] + 63) // 64
        indexing_jobs[doc_id] = {
            "status": "ready",
            "batches_done": batches_total,
            "batches_total": batches_total,
            "pages": stats["pages"],
            "chunks": stats["chunks"],
            "filename": filename,
            "error": None,
        }
        # Update hash cache
        cache = _load_hash_cache()
        cache[file_hash] = {
            "doc_id": doc_id,
            "filename": filename,
            "pages": stats["pages"],
            "chunks": stats["chunks"],
        }
        _save_hash_cache(cache)
    except Exception as exc:
        indexing_jobs[doc_id] = {
            "status": "error",
            "batches_done": 0,
            "batches_total": 0,
            "pages": 0,
            "chunks": 0,
            "filename": filename,
            "error": str(exc),
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the local FastEmbed model once at server startup
    config.get_embeddings()
    # Populate existing cached docs into indexing_jobs
    cache = _load_hash_cache()
    for h, entry in cache.items():
        did = entry.get("doc_id")
        if did:
            chunks = entry.get("chunks", 0)
            b_total = (chunks + 63) // 64 if chunks > 0 else 1
            indexing_jobs[did] = {
                "status": "ready",
                "batches_done": b_total,
                "batches_total": b_total,
                "pages": entry.get("pages", 0),
                "chunks": chunks,
                "filename": entry.get("filename", ""),
                "error": None,
            }
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
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. Validate file extension and MIME type
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .pdf files with application/pdf content type are accepted.",
        )

    # 2. Read content and enforce MAX_UPLOAD_MB
    content = await file.read()
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of {config.MAX_UPLOAD_MB}MB.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 3. SHA-256 caching: check if this file was already indexed
    file_hash = hashlib.sha256(content).hexdigest()
    cache = _load_hash_cache()

    if file_hash in cache:
        cached_entry = cache[file_hash]
        cached_doc_id = cached_entry.get("doc_id")
        try:
            client = chromadb.PersistentClient(path=config.DB_DIR)
            existing = [c.name for c in client.list_collections()]
            if cached_doc_id in existing:
                chunks = cached_entry.get("chunks", 0)
                b_total = (chunks + 63) // 64 if chunks > 0 else 1
                indexing_jobs[cached_doc_id] = {
                    "status": "ready",
                    "batches_done": b_total,
                    "batches_total": b_total,
                    "pages": cached_entry.get("pages", 0),
                    "chunks": chunks,
                    "filename": filename,
                    "error": None,
                }
                return {
                    "doc_id": cached_doc_id,
                    "status": "ready",
                    "filename": filename,
                    "pages": cached_entry.get("pages", 0),
                    "chunks": cached_entry.get("chunks", 0),
                    "cached": True,
                }
        except Exception:
            pass

    # 4. Cap uploads at 100 pages
    try:
        pdf_doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = pdf_doc.page_count
        pdf_doc.close()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF: {str(exc)}")

    if page_count > 100:
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {page_count} pages, which exceeds the maximum allowed limit of 100 pages.",
        )

    # 5. Non-blocking upload: save file and dispatch background task
    doc_id = uuid.uuid4().hex[:12]
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}_{filename}")
    with open(save_path, "wb") as f:
        f.write(content)

    indexing_jobs[doc_id] = {
        "status": "processing",
        "batches_done": 0,
        "batches_total": 0,
        "pages": page_count,
        "chunks": 0,
        "filename": filename,
        "error": None,
    }

    background_tasks.add_task(
        _background_indexer,
        save_path=save_path,
        doc_id=doc_id,
        file_hash=file_hash,
        filename=filename,
    )

    return {
        "doc_id": doc_id,
        "status": "processing",
        "filename": filename,
    }


@app.get("/status/{doc_id}")
def get_status(doc_id: str):
    job = indexing_jobs.get(doc_id)
    if not job:
        # Check Chroma collections directly as fallback
        try:
            client = chromadb.PersistentClient(path=config.DB_DIR)
            existing = [c.name for c in client.list_collections()]
            if doc_id in existing:
                return {
                    "status": "ready",
                    "batches_done": 1,
                    "batches_total": 1,
                    "pages": 0,
                    "chunks": 0,
                    "error": None,
                }
        except Exception:
            pass
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    return {
        "status": job["status"],
        "batches_done": job["batches_done"],
        "batches_total": job["batches_total"],
        "pages": job["pages"],
        "chunks": job["chunks"],
        "error": job["error"],
    }


@app.post("/ask")
def ask(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    doc_id = body.doc_id.strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id must not be empty.")

    # Check if doc_id is still being processed
    job = indexing_jobs.get(doc_id)
    if job:
        if job["status"] == "processing":
            raise HTTPException(status_code=409, detail="Still indexing")
        elif job["status"] == "error":
            raise HTTPException(status_code=500, detail=f"Indexing failed: {job['error']}")

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
