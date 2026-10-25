"""
ingest.py — Load, split, embed, and store any PDF into ChromaDB.
Usage as function: ingest_pdf(pdf_path, doc_id) -> {"pages": int, "chunks": int}
Usage as CLI: python ingest.py <file> [doc_id]
"""
import os
import re
import sys
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import config

BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 2
MAX_RETRIES = 5


def _extract_retry_seconds(error_msg: str) -> int:
    """Parse 'retry in X' or 'retry after X' from 429 error message, default 5s."""
    match = re.search(r"retry in (\d+)", str(error_msg), re.IGNORECASE)
    if match:
        return int(match.group(1)) + 2
    match_after = re.search(r"retry after (\d+)", str(error_msg), re.IGNORECASE)
    if match_after:
        return int(match_after.group(1)) + 2
    return 5


def ingest_pdf(pdf_path: str, doc_id: str) -> dict:
    """
    Ingest a PDF file into Chroma collection `doc_id`.
    Returns dict with page count and chunk count.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # Skip pages with no extractable text (e.g. scanned pages)
    valid_docs = []
    for doc in raw_docs:
        text = (doc.page_content or "").strip()
        if text:
            original_page = doc.metadata.get("page", 0)
            doc.metadata["page"] = int(original_page) + 1
            valid_docs.append(doc)

    if not valid_docs:
        raise ValueError("No readable text found (scanned PDF?)")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(valid_docs)
    if not chunks:
        raise ValueError("No readable text found (scanned PDF?)")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )

    db = Chroma(
        collection_name=doc_id,
        persist_directory=config.DB_DIR,
        embedding_function=embeddings,
    )

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                db.add_documents(batch)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = _extract_retry_seconds(err_str)
                    print(f"Batch {batch_num} 429 quota hit (attempt {attempt}). Waiting {wait}s...", flush=True)
                    time.sleep(wait)
                elif attempt < MAX_RETRIES:
                    wait = 2 * attempt
                    print(f"Batch {batch_num} error attempt {attempt}: {e}. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise

        if i + BATCH_SIZE < len(chunks):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return {
        "pages": len(valid_docs),
        "chunks": len(chunks),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_path> [doc_id]")
        sys.exit(1)

    file_path = sys.argv[1]
    doc_id = sys.argv[2] if len(sys.argv) > 2 else "default"
    print(f"Ingesting {file_path} into collection '{doc_id}'...")
    res = ingest_pdf(file_path, doc_id)
    print(f"Done! Pages: {res['pages']}, Chunks: {res['chunks']}")


if __name__ == "__main__":
    main()
