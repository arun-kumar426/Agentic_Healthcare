from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os

from .config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, GROQ_API_KEY

load_dotenv()


def get_llm():
    """Main LLM for all reasoning and answering."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.2,
        max_tokens=1024,
    )


def get_embeddings():
    """Embedding model for vector search."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
