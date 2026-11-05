"""Central settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", BASE_DIR / "artifacts"))
MPL_CACHE_DIR = Path(os.getenv("MPL_CACHE_DIR", BASE_DIR / ".mplcache"))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION = "documents"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 6                # chunks retrieved per question
MIN_SCORE = 0.25         # weak matches are dropped
MAX_CODE_ATTEMPTS = 3    # write + fix attempts for the code agent
SANDBOX_TIMEOUT = 40     # seconds (first run builds the matplotlib font cache)
