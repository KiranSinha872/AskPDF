import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Shared settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_DIR = "db"
UPLOAD_DIR = "uploads"
MAX_UPLOAD_MB = 20
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "minilm").lower().strip()
if EMBED_BACKEND in ("bge", "bge-small", "bge-small-en-v1.5"):
    EMBED_MODEL = "BAAI/bge-small-en-v1.5"
else:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 8  # Set to 8 for dense retrieval coverage

_shared_embeddings = None


def get_embeddings(model_name: str = None):
    """
    Singleton loader for local FastEmbed embeddings model.
    Defaults to EMBED_MODEL (sentence-transformers/all-MiniLM-L6-v2).
    Configured with batch_size=32 and threads=4 for optimal speed and memory balance.
    Uses passage_embed for embed_documents and query_embed for embed_query.
    """
    global _shared_embeddings
    if model_name is None:
        model_name = EMBED_MODEL

    if _shared_embeddings is None:
        from langchain_core.embeddings import Embeddings
        from fastembed import TextEmbedding

        threads = int(os.getenv("EMBED_THREADS", "4"))

        class FastEmbedWrapper(Embeddings):
            def __init__(self, model_name: str = model_name, batch_size: int = 32):
                self.batch_size = batch_size
                self.model = TextEmbedding(model_name=model_name, threads=threads)

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [
                    e.tolist()
                    for e in self.model.passage_embed(texts, batch_size=self.batch_size)
                ]

            def embed_query(self, text: str) -> List[float]:
                return next(self.model.query_embed(text)).tolist()

        _shared_embeddings = FastEmbedWrapper(model_name=model_name)
    return _shared_embeddings
