import datetime as dt
from pathlib import Path
from typing import Dict, Any

from .llm import get_llm

LOG_FILE = Path(__file__).resolve().parents[1] / "agent_logs.txt"


def log_interaction(
    user_query: str,
    answer: str,
    trace: Dict[str, Any] | None = None,
    eval_result: Dict[str, Any] | None = None,
) -> None:
    """Log interaction, trace and evaluation to a text file."""
    ts = dt.datetime.now().isoformat(timespec="seconds")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] USER: {user_query}\n")
        f.write(f"[{ts}] ANSWER: {answer}\n")
        if trace is not None:
            f.write(f"[{ts}] TRACE: {trace}\n")
        if eval_result is not None:
            f.write(f"[{ts}] EVAL: {eval_result}\n")
        f.write("\n")


def evaluate_answer(question: str, answer: str) -> Dict[str, Any]:
    """
    Simple LLM-based evaluator that scores:
    - correctness
    - relevance
    Returns a small JSON-like dict.
    """
    llm = get_llm()
    prompt = (
        "You are an evaluator. You will be given a question and an answer.\n"
        "Rate the answer on a scale of 1-5 for correctness and 1-5 for relevance.\n"
        "Then provide a one-sentence explanation.\n\n"
        "Return ONLY JSON with keys: correctness, relevance, explanation.\n\n"
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}\n"
    )
    raw = llm.invoke(prompt)
    text = raw.content if hasattr(raw, "content") else str(raw)
    # Try to parse JSON; fall back to plain text
    import json

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            pass
    return {
        "correctness": None,
        "relevance": None,
        "explanation": text[:400],
    }
