"""Validation rules for extracted action items."""

from __future__ import annotations

from datetime import datetime
from typing import List

from .extractor import ActionItem
from .utils import parse_date, similarity


class ActionItemValidator:
    """
    Applies a set of validation rules and annotates each item with issues.
    Also performs global checks (duplicates across the whole list).
    """

    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence

    def validate(self, items: List[ActionItem]) -> List[ActionItem]:
        """Validate and annotate a list of action items in place."""
        for item in items:
            item.issues = []
            self._check_missing_owner(item)
            self._check_missing_or_invalid_date(item)
            self._check_low_confidence(item)
            self._check_empty_task(item)

        # Global duplicate detection
        self._flag_duplicates(items)
        return items

    def _check_missing_owner(self, item: ActionItem) -> None:
        if not item.owner or not item.owner.strip():
            item.issues.append("missing_owner")

    def _check_missing_or_invalid_date(self, item: ActionItem) -> None:
        if not item.deadline:
            item.issues.append("missing_deadline")
            return
        # Try to re-parse to confirm validity
        parsed = parse_date(item.deadline)
        if parsed is None:
            item.issues.append("invalid_deadline")
        else:
            # Optional: flag past dates
            try:
                d = datetime.fromisoformat(parsed).date()
                if d < datetime.now().date():
                    item.issues.append("past_deadline")
            except Exception:
                pass

    def _check_low_confidence(self, item: ActionItem) -> None:
        if item.confidence < self.min_confidence:
            item.issues.append("low_confidence")

    def _check_empty_task(self, item: ActionItem) -> None:
        if not item.task or len(item.task.strip()) < 5:
            item.issues.append("empty_or_short_task")

    def _flag_duplicates(self, items: List[ActionItem]) -> None:
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i], items[j]
                same_owner = (
                    a.owner
                    and b.owner
                    and a.owner.lower() == b.owner.lower()
                )
                similar_task = similarity(a.task, b.task) > 82
                if same_owner and similar_task:
                    if "duplicate" not in a.issues:
                        a.issues.append("duplicate")
                    if "duplicate" not in b.issues:
                        b.issues.append("duplicate")

    def summary(self, items: List[ActionItem]) -> dict:
        """Return a quick quality summary."""
        total = len(items)
        if total == 0:
            return {"total": 0, "complete": 0, "issues": {}}

        complete = sum(
            1
            for i in items
            if i.owner and i.deadline and not i.issues
        )
        issue_counts: dict = {}
        for i in items:
            for iss in i.issues:
                issue_counts[iss] = issue_counts.get(iss, 0) + 1

        return {
            "total": total,
            "complete": complete,
            "incomplete": total - complete,
            "issues": issue_counts,
        }
