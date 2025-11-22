from langchain_groq import ChatGroq
from .config import GROQ_API_KEY, LLM_MODEL_NAME

def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.2
    )
