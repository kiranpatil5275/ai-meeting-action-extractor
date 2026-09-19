#!/usr/bin/env python3
"""
CLI evaluation script.
Usage:
    python evaluation/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extractor import ActionItemExtractor
from src.evaluator import ActionItemEvaluator


def main():
    transcripts_dir = ROOT / "data" / "sample_transcripts"
    annotations_path = ROOT / "data" / "annotations.json"

    print("=" * 60)
    print("AI Meeting Action-Item Extractor – Evaluation")
    print("=" * 60)
    print(f"Transcripts : {transcripts_dir}")
    print(f"Annotations : {annotations_path}")
    print()

    extractor = ActionItemExtractor(confidence_threshold=0.40)
    evaluator = ActionItemEvaluator(fuzzy_threshold=75)

    print("Running extraction + evaluation (first run downloads model)...")
    results = evaluator.evaluate_all(
        transcripts_dir=transcripts_dir,
        annotations_path=annotations_path,
        extractor=extractor,
    )

    print("\n--- Per-transcript results ---")
    for tid, m in results["per_transcript"].items():
        print(
            f"{tid}: gold={m['n_gold']:2d}  pred={m['n_predicted']:2d}  "
            f"exact_f1={m['exact_f1']:.3f}  fuzzy_f1={m['fuzzy_f1']:.3f}"
        )

    print("\n--- Macro averages ---")
    for k, v in results["macro"].items():
        print(f"  {k:20s}: {v:.3f}")

    print("\n--- Micro averages ---")
    for k, v in results["micro"].items():
        print(f"  {k:20s}: {v:.3f}")

    print("\n--- Totals ---")
    for k, v in results["totals"].items():
        print(f"  {k:20s}: {v}")

    # Save detailed results
    out_path = ROOT / "evaluation" / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
