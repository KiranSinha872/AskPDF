import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Shared settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_PATH = "data/GER-3620-P.pdf"
DB_DIR = "db"
CHAT_MODEL = "gemini-3.5-flash-lite"  # gemini-2.5-flash is 404 for new users on this key
EMBED_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 6
