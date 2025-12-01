#!/usr/bin/env python3
"""Utility helpers for building the summarization benchmark dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import plotly.graph_objects as go


@dataclass
class ModelRun:
    name: str
    summary_path: Path
    metrics: Dict[str, Dict[str, float]]


def collect_model_runs(root: Path) -> List[ModelRun]:
    runs: List[ModelRun] = []
    if not root.exists():
        return runs
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        summary_path = model_dir / "summary_metrics.json"
        if not summary_path.exists():
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        runs.append(ModelRun(name=model_dir.name, summary_path=summary_path, metrics=metrics))
    return runs


def _extract_metric(
    runs: Iterable[ModelRun],
    metric_name: str,
    stat: str = "avg",
) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    values: List[float] = []
    for run in runs:
        metric = run.metrics.get(metric_name)
        if not isinstance(metric, dict):
            continue
        value = metric.get(stat)
        if isinstance(value, (int, float)):
            labels.append(run.name)
            values.append(float(value))
    return labels, values


METRIC_SPECS = [
    (
        "Total Latency (ms)",
        "total_time_ms",
        "Average total end-to-end latency per article.",
    ),
    (
        "Prompt Throughput (tok/s)",
        "prompt_tokens_per_sec",
        "Average prompt-processing throughput derived from llama.cpp metrics.",
    ),
    (
        "Evaluation Throughput (tok/s)",
        "eval_tokens_per_sec",
        "Average generation throughput during completion.",
    ),
    (
        "Memory Usage (MB)",
        "memory_usage_mb",
        "Peak VRAM/DRAM usage captured by llama.cpp memory probes.",
    ),
    (
        "Summary Length (chars)",
        "summary_length_chars",
        "Average number of characters per generated summary.",
    ),
    (
        "F1",
        "f1",
        "Token-level F1 overlap with reference highlights.",
    ),
]


def _build_bar_chart(labels: List[str], values: List[float], title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color="#6366f1",
                hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="",
        template="plotly_white",
        margin=dict(t=60, r=20, b=40, l=60),
    )
    return fig


def build_graphs(runs: Iterable[ModelRun]) -> List[Tuple[str, str, go.Figure]]:
    bundles: List[Tuple[str, str, go.Figure]] = []
    cached_runs = list(runs)
    if not cached_runs:
        return bundles
    for title, metric_name, description in METRIC_SPECS:
        labels, values = _extract_metric(cached_runs, metric_name)
        if not labels:
            continue
        fig = _build_bar_chart(labels, values, title)
        fig.update_layout(legend_title="Model")
        fig.update_traces(marker_color="#2563eb")
        bundles.append((title, description, fig))
    return bundles

