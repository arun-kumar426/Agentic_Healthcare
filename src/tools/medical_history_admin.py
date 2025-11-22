from __future__ import annotations

from ..memory import add_patient_note


def add_or_update_history(
    patient_name: str,
    conditions: str,
    medications: str,
    free_text_note: str,
) -> str:
    """
    Store/update unstructured patient history (manual input from attendants).
    Saved into patient_notes.json via the memory module.
    """
    if not patient_name.strip():
        return "Patient name is required to add or update history."

    note = free_text_note.strip() or "No additional free-text note provided."

    add_patient_note(
        patient_name=patient_name,
        note=note,
        conditions=conditions,
        medications=medications,
    )

    return (
        f"✅ History updated for {patient_name}.\n"
        f"- Conditions: {conditions or 'N/A'}\n"
        f"- Medications: {medications or 'N/A'}\n"
        f"- Note: {note}"
    )
