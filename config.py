import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Shared settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_DIR = "db"
UPLOAD_DIR = "uploads"
MAX_UPLOAD_MB = 20
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 6
