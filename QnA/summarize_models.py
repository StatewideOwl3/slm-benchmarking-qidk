#!/usr/bin/env python3
"""Summarize parsed benchmark outputs for each model directory.

The script walks the immediate sub-directories of the given root (default: the
current working directory), looks for a ``parsed_outputs_*.json`` file, loads it,
computes aggregate statistics plus per-question time/memory traces, and writes a
``summary_metrics.json`` file back into the same directory. The resulting summary
is convenient for plotting cross-model graphs (latency, throughput, memory,
answer length, etc.).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

Number = Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize parsed benchmark outputs")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory that contains the model sub-folders (default: current directory)",
    )
    parser.add_argument(
        "--pattern",
        default="parsed_outputs_*.json",
        help="Glob pattern used to locate parsed output files inside each model folder",
    )
    parser.add_argument(
        "--output-name",
        default="summary_metrics.json",
        help="Filename for the generated summary stored inside each model folder",
    )
    return parser.parse_args()


def nested_get(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    if isinstance(current, (int, float)):
        return float(current)
    return None


def summarize_series(values: List[Optional[float]]) -> Optional[Dict[str, float]]:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    if not filtered:
        return None
    filtered.sort()
    return {
        "min": float(filtered[0]),
        "max": float(filtered[-1]),
        "avg": float(sum(filtered) / len(filtered)),
        "median": float(statistics.median(filtered)),
        "p90": float(filtered[int(0.9 * (len(filtered) - 1))]),
    }


def compute_throughput(entry: Dict[str, Any]) -> Optional[float]:
    total_tokens = entry.get("total_tokens")
    total_time_ms = entry.get("total_time_ms")
    if isinstance(total_tokens, (int, float)) and isinstance(total_time_ms, (int, float)) and total_time_ms > 0:
        return float(total_tokens) / (float(total_time_ms) / 1000.0)
    return None


SERIES_SPECS: Dict[str, Callable[[Dict[str, Any]], Optional[float]]] = {
    "total_time_ms": lambda e: nested_get(e, ("total_time_ms",)),
    "load_time_ms": lambda e: nested_get(e, ("load_time_ms",)),
    "prompt_eval_time_ms": lambda e: nested_get(e, ("prompt_eval_time", "total_prompt_eval_time_ms")),
    "prompt_eval_tokens": lambda e: nested_get(e, ("prompt_eval_time", "total_prompt_eval_tokens")),
    "prompt_tokens_per_sec": lambda e: nested_get(e, ("prompt_eval_time", "tokens_per_sec")),
    "eval_time_ms": lambda e: nested_get(e, ("eval_time", "total_eval_time_ms")),
    "eval_tokens": lambda e: nested_get(e, ("eval_time", "total_eval_tokens")),
    "eval_tokens_per_sec": lambda e: nested_get(e, ("eval_time", "tokens_per_sec")),
    "total_tokens": lambda e: nested_get(e, ("total_tokens",)),
    "throughput_tokens_per_sec": compute_throughput,
    "sampling_time_ms": lambda e: nested_get(e, ("sampling_time", "total_sampling_time_ms")),
    "sampling_runs": lambda e: nested_get(e, ("sampling_time", "total_sampling_runs")),
    "answer_length_chars": lambda e: float(len(e.get("model_answer", ""))),
    "memory_usage_mb": lambda e: nested_get(e, ("memory_usage_mb",)),
    "memory_avg_mb": lambda e: nested_get(e, ("memory_avg_mb",)),
    "memory_min_mb": lambda e: nested_get(e, ("memory_min_mb",)),
    "memory_max_mb": lambda e: nested_get(e, ("memory_max_mb",)),
    "graphs_reused": lambda e: nested_get(e, ("graphs_reused",)),
    "exact_match": lambda e: nested_get(e, ("exact_match",)),
    "f1": lambda e: nested_get(e, ("f1",)),
    "carbon_emissions_kg": lambda e: nested_get(e, ("carbon_emissions_kg",)),
}


def build_per_question(entry: Dict[str, Any]) -> Dict[str, Any]:
    prompt = entry.get("prompt_eval_time") or {}
    eval_section = entry.get("eval_time") or {}
    sampling = entry.get("sampling_time") or {}
    total_time_ms = entry.get("total_time_ms")
    total_tokens = entry.get("total_tokens")
    throughput = compute_throughput(entry)

    return {
        "id": entry.get("id"),
        "total_time_ms": total_time_ms,
        "load_time_ms": entry.get("load_time_ms"),
        "prompt_eval_time_ms": prompt.get("total_prompt_eval_time_ms"),
        "prompt_eval_tokens": prompt.get("total_prompt_eval_tokens"),
        "eval_time_ms": eval_section.get("total_eval_time_ms"),
        "eval_tokens": eval_section.get("total_eval_tokens"),
        "sampling_time_ms": sampling.get("total_sampling_time_ms"),
        "sampling_runs": sampling.get("total_sampling_runs"),
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": throughput,
        "memory_usage_mb": entry.get("memory_usage_mb"),
        "memory_avg_mb": entry.get("memory_avg_mb"),
        "memory_min_mb": entry.get("memory_min_mb"),
        "memory_max_mb": entry.get("memory_max_mb"),
        "answer_length_chars": len(entry.get("model_answer", "")),
        "graphs_reused": entry.get("graphs_reused"),
        "exact_match": entry.get("exact_match"),
        "f1": entry.get("f1"),
        "carbon_emissions_kg": entry.get("carbon_emissions_kg"),
    }


def summarize_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for label, extractor in SERIES_SPECS.items():
        series = [extractor(entry) for entry in entries]
        summary = summarize_series(series)
        if summary is not None:
            metrics[label] = summary

    per_question = [build_per_question(entry) for entry in entries]

    return {
        "question_count": len(entries),
        "metrics": metrics,
        "per_question": per_question,
    }


def find_parsed_file(model_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(model_dir.glob(pattern))
    return matches[0] if matches else None


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
        parsed_path = find_parsed_file(model_dir, args.pattern)
        if not parsed_path:
            continue
        with parsed_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list):
            print(f"Skipping {parsed_path} (expected a list)")
            continue

        summary = summarize_entries(entries)
        summary.update(
            {
                "model_directory": model_dir.name,
                "source_file": parsed_path.name,
                "root": str(model_dir.resolve()),
            }
        )

        output_path = model_dir / args.output_name
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
