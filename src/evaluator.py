"""Evaluation of extraction accuracy against gold annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from rapidfuzz import fuzz

from .extractor import ActionItem, ActionItemExtractor
from .validator import ActionItemValidator
from .utils import normalize_name, similarity


class ActionItemEvaluator:
    """
    Computes exact-match and fuzzy-match metrics.
    Primary key for matching: (task, owner)
    """

    def __init__(
        self,
        fuzzy_threshold: int = 75,
        exact_task: bool = False,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.exact_task = exact_task

    def evaluate_one(
        self,
        predicted: List[ActionItem],
        gold: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate a single transcript.
        Returns precision, recall, F1 for exact and fuzzy matching.
        """
        pred_pairs = [
            (normalize_name(p.owner or ""), p.task.strip().lower())
            for p in predicted
        ]
        gold_pairs = [
            (normalize_name(g.get("owner", "")), g.get("task", "").strip().lower())
            for g in gold
        ]

        # Exact match
        exact_matched = 0
        used_gold = set()
        for po, pt in pred_pairs:
            for gi, (go, gt) in enumerate(gold_pairs):
                if gi in used_gold:
                    continue
                if po == go and pt == gt:
                    exact_matched += 1
                    used_gold.add(gi)
                    break

        # Fuzzy match
        fuzzy_matched = 0
        used_gold_f = set()
        for po, pt in pred_pairs:
            for gi, (go, gt) in enumerate(gold_pairs):
                if gi in used_gold_f:
                    continue
                owner_ok = po == go or similarity(po, go) >= 90
                task_ok = (
                    pt == gt
                    or fuzz.token_sort_ratio(pt, gt) >= self.fuzzy_threshold
                )
                if owner_ok and task_ok:
                    fuzzy_matched += 1
                    used_gold_f.add(gi)
                    break

        n_pred = len(pred_pairs)
        n_gold = len(gold_pairs)

        def safe_div(a, b):
            return a / b if b else 0.0

        exact_p = safe_div(exact_matched, n_pred)
        exact_r = safe_div(exact_matched, n_gold)
        exact_f1 = (
            2 * exact_p * exact_r / (exact_p + exact_r)
            if (exact_p + exact_r)
            else 0.0
        )

        fuzzy_p = safe_div(fuzzy_matched, n_pred)
        fuzzy_r = safe_div(fuzzy_matched, n_gold)
        fuzzy_f1 = (
            2 * fuzzy_p * fuzzy_r / (fuzzy_p + fuzzy_r)
            if (fuzzy_p + fuzzy_r)
            else 0.0
        )

        return {
            "n_predicted": n_pred,
            "n_gold": n_gold,
            "exact_matched": exact_matched,
            "fuzzy_matched": fuzzy_matched,
            "exact_precision": round(exact_p, 3),
            "exact_recall": round(exact_r, 3),
            "exact_f1": round(exact_f1, 3),
            "fuzzy_precision": round(fuzzy_p, 3),
            "fuzzy_recall": round(fuzzy_r, 3),
            "fuzzy_f1": round(fuzzy_f1, 3),
        }

    def evaluate_all(
        self,
        transcripts_dir: str | Path,
        annotations_path: str | Path,
        extractor: ActionItemExtractor | None = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation over the full sample set.
        """
        transcripts_dir = Path(transcripts_dir)
        annotations = json.loads(Path(annotations_path).read_text())

        if extractor is None:
            extractor = ActionItemExtractor()
        validator = ActionItemValidator()

        results = {}
        aggregate = {
            "exact_matched": 0,
            "fuzzy_matched": 0,
            "n_predicted": 0,
            "n_gold": 0,
        }

        for tid, gold in annotations.items():
            fpath = transcripts_dir / f"{tid}.txt"
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            predicted = extractor.extract(text)
            predicted = validator.validate(predicted)
            metrics = self.evaluate_one(predicted, gold)
            results[tid] = metrics

            for k in aggregate:
                aggregate[k] += metrics[k]

        # Macro averages
        n = len(results) or 1
        macro = {
            "exact_precision": round(
                sum(r["exact_precision"] for r in results.values()) / n, 3
            ),
            "exact_recall": round(
                sum(r["exact_recall"] for r in results.values()) / n, 3
            ),
            "exact_f1": round(
                sum(r["exact_f1"] for r in results.values()) / n, 3
            ),
            "fuzzy_precision": round(
                sum(r["fuzzy_precision"] for r in results.values()) / n, 3
            ),
            "fuzzy_recall": round(
                sum(r["fuzzy_recall"] for r in results.values()) / n, 3
            ),
            "fuzzy_f1": round(
                sum(r["fuzzy_f1"] for r in results.values()) / n, 3
            ),
        }

        # Micro
        def safe(a, b):
            return round(a / b, 3) if b else 0.0

        micro = {
            "exact_precision": safe(aggregate["exact_matched"], aggregate["n_predicted"]),
            "exact_recall": safe(aggregate["exact_matched"], aggregate["n_gold"]),
            "fuzzy_precision": safe(aggregate["fuzzy_matched"], aggregate["n_predicted"]),
            "fuzzy_recall": safe(aggregate["fuzzy_matched"], aggregate["n_gold"]),
        }
        micro["exact_f1"] = (
            round(
                2
                * micro["exact_precision"]
                * micro["exact_recall"]
                / (micro["exact_precision"] + micro["exact_recall"] or 1),
                3,
            )
        )
        micro["fuzzy_f1"] = (
            round(
                2
                * micro["fuzzy_precision"]
                * micro["fuzzy_recall"]
                / (micro["fuzzy_precision"] + micro["fuzzy_recall"] or 1),
                3,
            )
        )

        return {
            "per_transcript": results,
            "macro": macro,
            "micro": micro,
            "totals": aggregate,
        }
