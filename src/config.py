import os
import streamlit as st

# App directories
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

PATIENTS_DIR = DATA_DIR / "patients"
DISEASES_DIR = DATA_DIR / "diseases"
APPOINTMENT_FILE = DATA_DIR / "records.xlsx"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"

# Get API key from Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Add it in Streamlit Secrets.")
