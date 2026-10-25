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
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 8  # Set to 8 for dense retrieval accuracy with bge-small

_shared_embeddings = None


def get_embeddings():
    """
    Singleton loader for local FastEmbed embeddings model.
    Loads 'BAAI/bge-small-en-v1.5' once into memory.
    """
    global _shared_embeddings
    if _shared_embeddings is None:
        from langchain_core.embeddings import Embeddings
        from fastembed import TextEmbedding

        class FastEmbedWrapper(Embeddings):
            def __init__(self, model_name: str = EMBED_MODEL):
                self.model = TextEmbedding(model_name=model_name)

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                # Embed texts in batches of 32
                results = []
                for i in range(0, len(texts), 32):
                    batch = texts[i : i + 32]
                    for emb in self.model.embed(batch, batch_size=32):
                        results.append(emb.tolist())
                return results

            def embed_query(self, text: str) -> List[float]:
                return next(self.model.embed([text])).tolist()

        _shared_embeddings = FastEmbedWrapper(model_name=EMBED_MODEL)
    return _shared_embeddings
