"""
server.py — FastAPI server for Document Q&A (any PDF upload & query).
Run: uvicorn server:app --port 8000
"""
import os
import pathlib
import uuid

import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from ingest import ingest_pdf
from rag_chain import build_chain

app = FastAPI(title="Document Q&A")

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

    return {
        "doc_id": doc_id,
        "filename": filename,
        "pages": stats["pages"],
        "chunks": stats["chunks"],
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
