#!/usr/bin/env python3
"""Evaluate summarization quality across all model directories.

This script mirrors ``QnA/evaluate.py`` but is tailored to the CNN/DailyMail-
style summarization task. It traverses the immediate subdirectories of the
specified root, searches for a ``parsed_outputs_*.json`` file produced by
``summarization.py``, and computes simple token-level F1 scores between the
model summaries and the reference ``highlights`` supplied in ``cnn_50.json``.

The aggregated results are written to ``evaluation_metrics.json`` within each
model folder and include optional CodeCarbon statistics when present in the
parsed outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

Summary = str
Summaries = Sequence[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate parsed summarization outputs against reference highlights")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory that contains the model sub-folders (default: current directory)",
    )
    parser.add_argument(
        "--pattern",
        default="parsed_outputs_*.json",
        help="Glob pattern to locate parsed outputs inside each model directory",
    )
    parser.add_argument(
        "--output-name",
        default="evaluation_metrics.json",
        help="Filename for the aggregated evaluation stored inside each model folder",
    )
    return parser.parse_args()


WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^0-9a-zA-Z]+")


def normalize_text(text: str) -> str:
    """Lowercase, trim whitespace, and strip punctuation for comparison."""

    lowered = text.lower().strip()
    no_punct = PUNCT_RE.sub(" ", lowered)
    collapsed = WHITESPACE_RE.sub(" ", no_punct)
    return collapsed.strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def overlap_counts(pred_tokens: List[str], ref_tokens: List[str]) -> int:
    ref_freq: Dict[str, int] = {}
    for token in ref_tokens:
        ref_freq[token] = ref_freq.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if ref_freq.get(token, 0) > 0:
            overlap += 1
            ref_freq[token] -= 1
    return overlap


def safe_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = overlap_counts(pred_tokens, ref_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: Summary, highlights: Summaries) -> float:
    """Return the best token-level F1 over all provided reference highlights."""

    pred_tokens = tokenize(prediction)
    ref_token_lists = [tokenize(ref) for ref in highlights]
    if not ref_token_lists:
        return float(not pred_tokens)
    scores = [safe_f1(pred_tokens, tokens) for tokens in ref_token_lists]
    return max(scores)


def load_entries(parsed_path: Path) -> List[Dict[str, Any]]:
    with parsed_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {parsed_path}")
    return data


def evaluate_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    f1_scores: List[float] = []
    carbon_values: List[float] = []

    per_sample: List[Dict[str, Any]] = []

    for entry in entries:
        model_summary = entry.get("model_answer", "")
        highlights = entry.get("highlights", [])
        if not isinstance(highlights, list):
            highlights = []

        f1 = best_f1(model_summary, highlights)
        carbon = entry.get("carbon_emissions_kg")

        f1_scores.append(f1)
        if isinstance(carbon, (int, float)):
            carbon_values.append(float(carbon))

        entry["f1"] = f1

        per_sample.append(
            {
                "id": entry.get("id"),
                "article": entry.get("article"),
                "model_summary": model_summary,
                "highlights": highlights,
                "f1": f1,
                "carbon_emissions_kg": carbon,
            }
        )

    overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    total_carbon = sum(carbon_values) if carbon_values else None
    avg_carbon = (total_carbon / len(carbon_values)) if carbon_values else None

    return {
        "overall": {
            "f1": overall_f1,
            "sample_count": len(entries),
            **(
                {
                    "carbon_emissions_kg_sum": total_carbon,
                    "carbon_emissions_kg_avg": avg_carbon,
                }
                if total_carbon is not None and avg_carbon is not None
                else {}
            ),
        },
        "per_sample": per_sample,
    }


def find_parsed_file(model_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(model_dir.glob(pattern))
    return matches[0] if matches else None


def evaluate_directory(model_dir: Path, pattern: str, output_name: str) -> None:
    parsed_path = find_parsed_file(model_dir, pattern)
    if not parsed_path:
        return
    entries = load_entries(parsed_path)
    results = evaluate_entries(entries)
    with parsed_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)
    results.update(
        {
            "model_directory": model_dir.name,
            "source_file": parsed_path.name,
            "root": str(model_dir.resolve()),
        }
    )

    output_path = model_dir / output_name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved evaluation to {output_path}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    model_dirs = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')])
    if not model_dirs:
        print(f"No sub-directories found under {root}")
        return

    for model_dir in model_dirs:
        try:
            evaluate_directory(model_dir, args.pattern, args.output_name)
        except Exception as err:  # noqa: BLE001
            print(f"Skipping {model_dir}: {err}")


if __name__ == "__main__":
    main()


