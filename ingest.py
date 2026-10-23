"""
ingest.py — Load, split, embed, and store GER-3620-P.pdf into ChromaDB.
Usage: python ingest.py
"""
import os
import re
import time
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import config

BATCH_SIZE = 50
# Free tier: 100 embed requests / minute. Each batch = 50 docs but the SDK
# sends them as individual calls, so 50 calls/batch. Wait 65s between batches
# to stay safely under the 100 RPM limit.
SLEEP_BETWEEN_BATCHES = 65  # seconds
MAX_RETRIES = 5


def _extract_retry_seconds(error_msg: str) -> int:
    """Parse 'retry in X.Xs' from the 429 error message, default 60s."""
    match = re.search(r"retry in (\d+)", str(error_msg), re.IGNORECASE)
    return int(match.group(1)) + 5 if match else 65


def main():
    print(f"Loading PDF from: {config.PDF_PATH}")
    loader = PyPDFLoader(config.PDF_PATH)
    raw_docs = loader.load()
    print(f"Pages loaded: {len(raw_docs)}")

    # Fix page metadata: make it 1-based
    for doc in raw_docs:
        original_page = doc.metadata.get("page", 0)
        doc.metadata["page"] = int(original_page) + 1

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Chunks created: {len(chunks)}")

    # Delete existing db and rebuild
    if os.path.exists(config.DB_DIR):
        print(f"Deleting existing db at: {config.DB_DIR}")
        shutil.rmtree(config.DB_DIR)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding {len(chunks)} chunks in {total_batches} batches of {BATCH_SIZE}...")
    print(f"(~{SLEEP_BETWEEN_BATCHES}s sleep between batches to respect free-tier 100 RPM quota)")

    db = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", flush=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if db is None:
                    db = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=config.DB_DIR,
                    )
                else:
                    db.add_documents(batch)
                print(f"    Batch {batch_num} OK", flush=True)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = _extract_retry_seconds(err_str)
                    print(f"    429 quota hit (attempt {attempt}). Waiting {wait}s...", flush=True)
                    time.sleep(wait)
                elif attempt < MAX_RETRIES:
                    wait = 10 * attempt
                    print(f"    Error attempt {attempt}: {e}. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise

        # Sleep between batches (not after the last one)
        if i + BATCH_SIZE < len(chunks):
            print(f"    Sleeping {SLEEP_BETWEEN_BATCHES}s before next batch...", flush=True)
            time.sleep(SLEEP_BETWEEN_BATCHES)

    count = db._collection.count()
    print(f"\nDone! ChromaDB stored at: {os.path.abspath(config.DB_DIR)}")
    print(f"Total documents in db: {count}")


if __name__ == "__main__":
    main()
