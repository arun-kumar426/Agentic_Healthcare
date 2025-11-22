import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PATIENTS_DIR = DATA_DIR / "patients"
DISEASES_DIR = DATA_DIR / "diseases"
APPOINTMENT_FILE = DATA_DIR / "records.xlsx"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model (LLama 3)
LLM_MODEL_NAME = "llama-3.1-8b-instant"

# Read key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise RuntimeError(
        "GROQ_API_KEY not set. Add it in your .env file like:\n"
        "GROQ_API_KEY=gsk_your_key_here"
    )
