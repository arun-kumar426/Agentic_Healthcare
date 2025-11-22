from __future__ import annotations

from pathlib import Path
from typing import Optional, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llm import get_llm, get_embeddings
from ..config import DISEASES_DIR

_vectorstore: Optional[FAISS] = None


def _load_disease_docs():
    docs = []
    if not DISEASES_DIR.exists():
        return docs

    for path in DISEASES_DIR.iterdir():
        if path.suffix.lower() in [".pdf"]:
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
    return docs


def _get_or_build_vectorstore() -> Optional[FAISS]:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    docs = _load_disease_docs()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    _vectorstore = FAISS.from_documents(chunks, embeddings)
    return _vectorstore


def get_disease_information(disease_query: str) -> str:
    """
    Provide disease/condition information using:
    - Local WHO/Medline docs in data/diseases (RAG)
    - Fallback to LLM-only explanation if no docs.
    """
    llm = get_llm()
    vs = _get_or_build_vectorstore()

    if vs is None:
        # Fallback: no local docs available
        prompt = ChatPromptTemplate.from_template(
            """You are a cautious medical information assistant.
User query: {query}

TASK:
1. Explain the disease/condition in simple terms.
2. Summarize causes, symptoms, diagnosis, and standard treatments.
3. Mention red-flag symptoms for emergency care.
4. End with: "This is not a medical diagnosis. Please consult a licensed doctor."

Keep the answer under 300 words and avoid giving exact drug doses."""
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"query": disease_query})

    # Use RAG over disease docs
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    docs: List = retriever.invoke(disease_query)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(
        """You are a medical information assistant using WHO/Medline style content.

CONTEXT:
{context}

USER QUESTION:
{query}

TASK:
1. Answer the question using the above context.
2. If the answer is not fully in the context, say what is missing.
3. Provide causes, symptoms, diagnosis, and standard treatments.
4. Mention red-flag symptoms for emergency care.
5. End with: "This is not a medical diagnosis. Please consult a licensed doctor."

Keep answer under 350 words and avoid exact medication doses."""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "query": disease_query})
