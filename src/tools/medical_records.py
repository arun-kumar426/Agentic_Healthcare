from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llm import get_llm, get_embeddings
from ..config import PATIENTS_DIR
from ..memory import get_patient_context, get_patient_notes, save_patient_summary


# Map patient names to the PDF files you have
PATIENT_FILES: Dict[str, List[str]] = {
    "rebecca nagle": ["sample_patient.pdf"],
    "anjali mehra": ["sample_report_anjali.pdf"],
    "david thompson": ["sample_report_david.pdf"],
    "ramesh kulkarni": ["sample_report_ramesh.pdf"],
}


def _load_patient_docs(patient_name: str):
    """
    Load one or more PDF files mapped to a patient.
    """
    key = patient_name.lower().strip()
    if key not in PATIENT_FILES:
        raise ValueError(
            f"No documents configured for patient '{patient_name}'. "
            "Update PATIENT_FILES in medical_records.py."
        )

    docs = []
    for fname in PATIENT_FILES[key]:
        pdf_path = PATIENTS_DIR / fname
        if not pdf_path.exists():
            raise FileNotFoundError(f"Patient PDF not found: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    return docs


def _build_vectorstore(docs):
    """
    Build vector search index (FAISS) from PDF chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def summarize_patient_history(patient_name: str, question: str | None = None) -> str:
    """
    Summarize patient medical history, combining:
    - EHR PDFs (RAG)
    - Stored memory summaries
    - Manually added notes
    """
    llm = get_llm()
    docs = _load_patient_docs(patient_name)
    vs = _build_vectorstore(docs)

    if question is None:
        question = (
            "Provide a concise summary of this patient's medical history, "
            "key diagnoses, treatments, medications, and recent encounters."
        )

    retriever = vs.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)

    ehr_context = "\n\n".join(d.page_content for d in relevant_docs)

    memory_context = get_patient_context(patient_name)
    notes_context = get_patient_notes(patient_name)

    full_context = "\n\n=== EHR DOCUMENTS ===\n\n" + ehr_context
    if memory_context:
        full_context += "\n\n=== PREVIOUS SUMMARIES (MEMORY) ===\n\n" + memory_context
    if notes_context:
        full_context += "\n\n=== MANUAL NOTES ===\n\n" + notes_context

    prompt = ChatPromptTemplate.from_template(
        """
You are a clinical assistant.
You will receive EHR context, past summaries, manual notes and a question.

CONTEXT:
{context}

QUESTION:
{question}

TASK:
1. Provide a clear, structured clinical summary.
2. Highlight diagnoses, medications, vitals, tests, and follow-up plans.
3. If there are conflicts, mention them explicitly.
4. Keep response under 250 words.
5. Write in simple, readable clinical language.

Answer:
"""
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"context": full_context, "question": question})

    # Save to long-term memory
    save_patient_summary(patient_name, result, source="ehr_summary")

    return result
