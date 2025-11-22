from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

from .config import DATA_DIR

MEMORY_FILE = DATA_DIR / "patient_memory.json"
NOTES_FILE = DATA_DIR / "patient_notes.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------- PATIENT MEMORY (summaries from RAG) ----------

def save_patient_summary(patient_name: str, summary: str, source: str = "rag") -> None:
    data = _load_json(MEMORY_FILE)
    key = patient_name.lower().strip()
    entry = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "summary": summary,
    }
    data.setdefault(key, []).append(entry)
    _save_json(MEMORY_FILE, data)


def get_patient_context(patient_name: str, max_entries: int = 5) -> str:
    data = _load_json(MEMORY_FILE)
    key = patient_name.lower().strip()
    entries: List[Dict[str, Any]] = data.get(key, [])
    if not entries:
        return ""

    recent = entries[-max_entries:]
    chunks = []
    for e in recent:
        chunks.append(
            f"[{e['timestamp']} from {e['source']}]\n{e['summary']}"
        )
    return "\n\n".join(chunks)


# ---------- PATIENT NOTES (manual updates from UI) ----------

def add_patient_note(
    patient_name: str,
    note: str,
    conditions: str = "",
    medications: str = "",
) -> None:
    data = _load_json(NOTES_FILE)
    key = patient_name.lower().strip()
    entry = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "conditions": conditions,
        "medications": medications,
    }
    data.setdefault(key, []).append(entry)
    _save_json(NOTES_FILE, data)


def get_patient_notes(patient_name: str, max_entries: int = 10) -> str:
    data = _load_json(NOTES_FILE)
    key = patient_name.lower().strip()
    entries: List[Dict[str, Any]] = data.get(key, [])
    if not entries:
        return ""
    recent = entries[-max_entries:]
    chunks = []
    for e in recent:
        chunks.append(
            f"[{e['timestamp']}] Conditions: {e['conditions']} | "
            f"Medications: {e['medications']}\nNote: {e['note']}"
        )
    return "\n\n".join(chunks)
