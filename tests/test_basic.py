"""Basic smoke tests (no model download required for most)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaner import TranscriptCleaner
from src.utils import parse_date, normalize_name, clean_task_text
from src.validator import ActionItemValidator
from src.extractor import ActionItem


def test_cleaner_segment():
    text = """Sarah Chen: Let's prepare the budget.
Mike Torres: I will schedule the meeting by Friday."""
    cleaner = TranscriptCleaner()
    utts = cleaner.segment(text)
    assert len(utts) >= 2
    assert utts[0].speaker == "Sarah Chen"
    assert "budget" in utts[0].text.lower()


def test_parse_date():
    assert parse_date("September 30th") is not None
    assert parse_date("next Friday") is not None
    assert parse_date("by the end of the month") is not None


def test_normalize_name():
    assert normalize_name("  sarah chen  ") == "Sarah Chen"
    assert normalize_name("Mr. John Doe") == "John Doe"


def test_validator_missing_owner():
    item = ActionItem(task="Do something", owner=None, deadline="2024-10-01", confidence=0.8)
    validator = ActionItemValidator()
    validator.validate([item])
    assert "missing_owner" in item.issues


def test_clean_task():
    t = clean_task_text("can you please prepare the report by Friday")
    assert t.lower().startswith("prepare")


if __name__ == "__main__":
    test_cleaner_segment()
    test_parse_date()
    test_normalize_name()
    test_validator_missing_owner()
    test_clean_task()
    print("All basic tests passed.")
