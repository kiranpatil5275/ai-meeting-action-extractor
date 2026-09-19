"""Utility helpers for dates, names, and fuzzy matching."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

import dateparser
from rapidfuzz import fuzz


def normalize_name(name: str) -> str:
    """Normalize a person name for comparison."""
    if not name:
        return ""
    name = re.sub(r"\s+", " ", name.strip())
    # Remove common titles
    name = re.sub(r"^(mr|mrs|ms|dr|prof)\.?\s+", "", name, flags=re.I)
    return name.title()


def parse_date(text: str, reference: Optional[datetime] = None) -> Optional[str]:
    """
    Parse a date expression into ISO format (YYYY-MM-DD).
    Handles absolute and relative dates.
    Returns None if unparseable.
    """
    if not text or not text.strip():
        return None

    reference = reference or datetime.now()
    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": reference,
        "RETURN_AS_TIMEZONE_AWARE": False,
    }

    # Clean common meeting phrases
    cleaned = re.sub(
        r"\b(by|before|until|due|deadline|on|for)\b",
        " ",
        text,
        flags=re.I,
    ).strip()

    parsed = dateparser.parse(cleaned, settings=settings)
    if parsed is None:
        # Try a few manual relative patterns
        lower = cleaned.lower()
        today = reference.date()
        if "end of the month" in lower or "eom" in lower:
            next_month = today.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            return last_day.isoformat()
        if "next week" in lower:
            return (today + timedelta(days=7)).isoformat()
        if "this week" in lower:
            return (today + timedelta(days=3)).isoformat()
        return None

    return parsed.date().isoformat()


def fuzzy_match(a: str, b: str, threshold: int = 80) -> bool:
    """Return True if two strings are similar enough."""
    if not a or not b:
        return False
    score = fuzz.token_sort_ratio(a.lower(), b.lower())
    return score >= threshold


def similarity(a: str, b: str) -> float:
    """Return similarity score 0–100."""
    if not a or not b:
        return 0.0
    return float(fuzz.token_sort_ratio(a.lower(), b.lower()))


def extract_person_candidates(text: str) -> list[str]:
    """Simple heuristic person name extractor (fallback)."""
    # Capitalized words that look like names
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    return list(dict.fromkeys(re.findall(pattern, text)))


def clean_task_text(text: str) -> str:
    """Normalize task description."""
    text = re.sub(r"\s+", " ", text.strip())
    # Remove leading verbs that are pure filler
    text = re.sub(
        r"^(please|can you|could you|I need you to|you should|let's)\s+",
        "",
        text,
        flags=re.I,
    )
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    return text
