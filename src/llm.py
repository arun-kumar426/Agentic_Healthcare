from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from .config import GROQ_API_KEY, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME

def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.2
    )

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
