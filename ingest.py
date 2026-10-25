"""
ingest.py — Load, split, embed, and store any PDF into ChromaDB using local FastEmbed.
Usage as function: ingest_pdf(pdf_path, doc_id) -> {"pages": int, "chunks": int}
Usage as CLI: python ingest.py <file> [doc_id]
"""
import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config


def ingest_pdf(pdf_path: str, doc_id: str) -> dict:
    """
    Ingest a PDF file into Chroma collection `doc_id` using FastEmbed.
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

    # Load shared embeddings singleton
    embeddings = config.get_embeddings()

    db = Chroma(
        collection_name=doc_id,
        persist_directory=config.DB_DIR,
        embedding_function=embeddings,
    )

    # Add all chunks to Chroma directly without sleeps or retry loops
    db.add_documents(chunks)

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
    print(f"Ingesting {file_path} into collection '{doc_id}' with FastEmbed...")
    res = ingest_pdf(file_path, doc_id)
    print(f"Done! Pages: {res['pages']}, Chunks: {res['chunks']}")


if __name__ == "__main__":
    main()
