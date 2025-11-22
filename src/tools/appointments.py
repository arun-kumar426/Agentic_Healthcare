from __future__ import annotations

import pandas as pd
from typing import List, Dict, Any, Optional

from ..config import APPOINTMENT_FILE


def _load_appointments() -> pd.DataFrame:
    if not APPOINTMENT_FILE.exists():
        df = pd.DataFrame(
            columns=[
                "appointment_id",
                "patient_name",
                "doctor_name",
                "speciality",
                "date",
                "time_slot",
                "status",
            ]
        )
        df.to_excel(APPOINTMENT_FILE, index=False)
        return df

    return pd.read_excel(APPOINTMENT_FILE)


def _save_appointments(df: pd.DataFrame) -> None:
    df.to_excel(APPOINTMENT_FILE, index=False)


def list_available_slots(
    speciality: Optional[str] = None,
    date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return available appointment slots filtered by speciality and/or date."""
    df = _load_appointments()
    if df.empty:
        return []

    if "status" not in df.columns:
        # Fail gracefully instead of crashing
        return []

    mask = df["status"].astype(str).str.lower() == "available"

    if speciality:
        if "speciality" in df.columns:
            mask &= df["speciality"].astype(str).str.lower() == speciality.lower()

    if date:
        if "date" in df.columns:
            mask &= df["date"].astype(str) == str(date)

    available = df[mask]
    return available.to_dict(orient="records")


def book_appointment(
    patient_name: str,
    reason: str,
    speciality: str,
    preferred_date: Optional[str] = None,
) -> str:
    """Book the first available slot for a given speciality and optional date."""
    df = _load_appointments()
    if df.empty:
        return "No appointment data found in records.xlsx. Please add some rows first."

    if "status" not in df.columns or "speciality" not in df.columns:
        return (
            "records.xlsx is missing required columns: 'status' and/or 'speciality'. "
            "Please include them and try again."
        )

    mask = df["status"].astype(str).str.lower() == "available"
    mask &= df["speciality"].astype(str).str.lower() == speciality.lower()

    if preferred_date and "date" in df.columns:
        mask &= df["date"].astype(str) == str(preferred_date)

    available = df[mask]

    if available.empty:
        return f"No available slots found for speciality '{speciality}' on {preferred_date or 'any date'}."

    available = available.sort_values(by=["date", "time_slot"])
    row_index = available.index[0]

    df.at[row_index, "patient_name"] = patient_name
    df.at[row_index, "status"] = "booked"
    _save_appointments(df)

    booked_row = df.loc[row_index]
    return (
        "✅ Appointment booked!\n"
        f"Patient: {patient_name}\n"
        f"Doctor: {booked_row.get('doctor_name', 'N/A')} "
        f"({booked_row.get('speciality', 'N/A')})\n"
        f"Date: {booked_row.get('date', 'N/A')}\n"
        f"Time: {booked_row.get('time_slot', 'N/A')}\n"
        f"Reason: {reason}"
    )
