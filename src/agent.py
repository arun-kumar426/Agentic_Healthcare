from __future__ import annotations

import json
from typing import Literal, TypedDict, Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_llm
from .tools.appointments import book_appointment, list_available_slots
from .tools.medical_records import summarize_patient_history
from .tools.disease_info import get_disease_information
from .tools.medical_history_admin import add_or_update_history
from .memory import get_patient_context


class Plan(TypedDict, total=False):
    task_type: Literal["BOOK_APPOINTMENT", "PATIENT_SUMMARY",
                       "DISEASE_INFO", "UPDATE_HISTORY"]
    patient_name: Optional[str]
    reason: Optional[str]
    speciality: Optional[str]
    date: Optional[str]
    disease: Optional[str]
    conditions: Optional[str]
    medications: Optional[str]
    note: Optional[str]


def _plan_from_query(user_query: str) -> Plan:
    """
    Use LLM as a planner to decompose the user's intent.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """You are a planning agent for a healthcare assistant.

Your job is to analyse the user's message and decide:

- Is it about booking an appointment?
- Is it about summarizing a patient's medical history?
- Is it about getting information about a disease/condition?
- Is it about updating a patient's medical history (conditions/medications/notes)?

Return ONLY valid JSON with these keys:
- "task_type": one of "BOOK_APPOINTMENT", "PATIENT_SUMMARY", "DISEASE_INFO", "UPDATE_HISTORY"
- "patient_name": full patient name if mentioned, else null
- "reason": short reason for visit if mentioned, else null
- "speciality": e.g. "nephrologist", "cardiologist", else null
- "date": preferred date in YYYY-MM-DD if explicitly mentioned, else null
- "disease": disease name or question for disease info, else null
- "conditions": chronic conditions to store/update, else null
- "medications": important medications, else null
- "note": free-text note to store/update, else null

User message:
{query}
"""
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": user_query})

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1:
        raw = raw[first_brace : last_brace + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "task_type": "DISEASE_INFO",
            "patient_name": None,
            "reason": None,
            "speciality": None,
            "date": None,
            "disease": user_query,
            "conditions": None,
            "medications": None,
            "note": None,
        }
    return data  # type: ignore[return-value]


def run_agent(user_query: str) -> Dict[str, Any]:
    """
    Main entry: take user query, plan, call the right tool,
    and return both the final answer and a detailed trace.
    """
    plan = _plan_from_query(user_query)
    task = plan.get("task_type")
    tool_name = ""
    tool_input: Dict[str, Any] = {}
    tool_output: str = ""
    patient_context_used = ""

    if task == "BOOK_APPOINTMENT":
        tool_name = "book_appointment"
        patient_name = plan.get("patient_name") or "Unknown Patient"
        reason = plan.get("reason") or user_query
        speciality = plan.get("speciality") or "general physician"
        date = plan.get("date")

        tool_input = {
            "patient_name": patient_name,
            "reason": reason,
            "speciality": speciality,
            "preferred_date": date,
        }
        tool_output = book_appointment(**tool_input)

        final_answer = tool_output

    elif task == "PATIENT_SUMMARY":
        tool_name = "summarize_patient_history"
        patient_name = plan.get("patient_name")
        if not patient_name:
            final_answer = (
                "I need a patient name to summarize the medical history. "
                "For example: 'Summarize history for Anjali Mehra.'"
            )
        else:
            patient_context_used = get_patient_context(patient_name)
            tool_input = {"patient_name": patient_name}
            tool_output = summarize_patient_history(patient_name)
            final_answer = tool_output

    elif task == "UPDATE_HISTORY":
        tool_name = "add_or_update_history"
        patient_name = plan.get("patient_name")
        if not patient_name:
            final_answer = "Please specify the patient's full name to update their history."
        else:
            tool_input = {
                "patient_name": patient_name,
                "conditions": plan.get("conditions") or "",
                "medications": plan.get("medications") or "",
                "free_text_note": plan.get("note") or user_query,
            }
            tool_output = add_or_update_history(**tool_input)
            final_answer = tool_output

    else:  # "DISEASE_INFO" or fallback
        tool_name = "get_disease_information"
        disease_query = plan.get("disease") or user_query
        tool_input = {"disease_query": disease_query}
        tool_output = get_disease_information(disease_query)
        final_answer = tool_output

    trace: Dict[str, Any] = {
        "user_query": user_query,
        "plan": plan,
        "selected_tool": tool_name,
        "tool_input": tool_input,
        "tool_output_preview": tool_output[:400],
        "patient_memory_used": patient_context_used[:400] if patient_context_used else "",
    }

    return {
        "answer": final_answer,
        "trace": trace,
    }
