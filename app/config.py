"""Configuration constants for the research assistant service."""
from pathlib import Path

# Base directories
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "indexes"

# File paths for vector store artifacts
FAISS_INDEX_PATH = INDEX_DIR / "default.index"
METADATA_PATH = INDEX_DIR / "default_metadata.json"

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Google Drive configuration placeholders
# Update this path to point to your service account JSON credentials file.
GOOGLE_SERVICE_ACCOUNT_FILE = Path("config/service_account.json")
GOOGLE_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Ensure directories exist when the module is imported.
for directory in (DATA_DIR, RAW_DATA_DIR, INDEX_DIR):
    directory.mkdir(parents=True, exist_ok=True)
