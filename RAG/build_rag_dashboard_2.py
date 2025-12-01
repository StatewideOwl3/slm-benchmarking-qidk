#!/usr/bin/env python3
"""Visualization toolkit for RAG benchmark runs with per-stage carbon metrics.

This variant focuses on ``*-ragres2`` folders that include the updated
``rag_benchmark.py`` outputs (per-question retrieval/generation carbon). It loads
summary statistics plus parsed outputs, aggregates the additional metrics, and
renders insight-driven Plotly charts for quick comparisons between models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.io as pio


@dataclass
class ModelReport:
    label: str
    retrieval_latency_ms: Optional[float]
    load_ms: Optional[float]
    prompt_ms: Optional[float]
    eval_ms: Optional[float]
    throughput_eval_tps: Optional[float]
    eval_tokens_avg: Optional[float]
    memory_mb: Optional[float]
    semantic_similarity: Optional[float]
    f1: Optional[float]
    retrieval_carbon_avg: Optional[float]
    generation_carbon_avg: Optional[float]
    question_count: int

    @property
    def blended_accuracy(self) -> Optional[float]:
        if self.f1 is None or self.semantic_similarity is None:
            return None
        return 0.5 * (self.f1 + self.semantic_similarity)

    @property
    def generation_latency_ms(self) -> Optional[float]:
        components = [self.load_ms, self.prompt_ms, self.eval_ms]
        if any(component is None for component in components):
            return None
        return sum(component for component in components if component is not None)

    @property
    def total_latency_ms(self) -> Optional[float]:
        generation = self.generation_latency_ms
        if self.retrieval_latency_ms is None or generation is None:
            return None
        return self.retrieval_latency_ms + generation

    @property
    def carbon_ratio(self) -> Optional[float]:
        if not self.generation_carbon_avg or not self.retrieval_carbon_avg:
            return None
        if self.generation_carbon_avg == 0:
            return None
        return self.retrieval_carbon_avg / self.generation_carbon_avg

    @property
    def total_carbon_per_question(self) -> Optional[float]:
        if self.retrieval_carbon_avg is None or self.generation_carbon_avg is None:
            return None
        return self.retrieval_carbon_avg + self.generation_carbon_avg

    @property
    def generation_carbon_per_100_tokens(self) -> Optional[float]:
        if self.generation_carbon_avg is None or not self.eval_tokens_avg:
            return None
        if self.eval_tokens_avg == 0:
            return None
        return (self.generation_carbon_avg / self.eval_tokens_avg) * 100.0


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG dashboard (ragres2)")
    parser.add_argument(
        "--rag-root",
        default=".",
        help="Root containing *-ragres2 folders",
    )
    parser.add_argument(
        "--output",
        default="rag_dashboard_2.html",
        help="Path to the generated HTML dashboard",
    )
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(container: Dict[str, object], key: str) -> Optional[float]:
    value = container.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _metric_avg(metrics_root: Dict[str, object], metric_name: str) -> Optional[float]:
    bucket = metrics_root.get(metric_name)
    if isinstance(bucket, dict):
        avg = bucket.get("avg")
        if isinstance(avg, (int, float)):
            return float(avg)
    return None


def _find_parsed_file(model_dir: Path) -> Optional[Path]:
    matches = sorted(model_dir.glob("parsed_outputs_*.json"))
    return matches[0] if matches else None


def _aggregate_carbon(parsed_path: Path) -> Tuple[Optional[float], Optional[float], int]:
    data = _load_json(parsed_path)
    if not isinstance(data, list) or not data:
        return None, None, 0
    retrieval_values: List[float] = []
    generation_values: List[float] = []
    for entry in data:
        rc = entry.get("retrieval_carbon_kg")
        gc = entry.get("generation_carbon_kg") or entry.get("carbon_emissions_kg")
        if isinstance(rc, (int, float)):
            retrieval_values.append(float(rc))
        if isinstance(gc, (int, float)):
            generation_values.append(float(gc))
    retrieval_avg = mean(retrieval_values) if retrieval_values else None
    generation_avg = mean(generation_values) if generation_values else None
    return retrieval_avg, generation_avg, len(data)


def collect_reports(root: Path) -> List[ModelReport]:
    summary_paths = sorted(root.glob("*ragres2*/*/summary.json"))
    reports: List[ModelReport] = []
    for summary_path in summary_paths:
        model_dir = summary_path.parent
        summary_metrics_path = model_dir / "summary_metrics.json"
        if not summary_metrics_path.exists():
            continue
        parsed_path = _find_parsed_file(model_dir)
        if not parsed_path:
            continue

        summary_payload = _load_json(summary_path)
        metrics_payload = _load_json(summary_metrics_path)
        metrics_root = metrics_payload.get("metrics") or {}
        retrieval_block = summary_payload.get("retrieval_latency_ms") or {}
        evaluation_block = summary_payload.get("evaluation") or {}

        retrieval_carbon_avg, generation_carbon_avg, question_count = _aggregate_carbon(parsed_path)
        if question_count == 0:
            continue

        report = ModelReport(
            label=model_dir.parent.name,
            retrieval_latency_ms=_safe_float(retrieval_block, "avg"),
            load_ms=_metric_avg(metrics_root, "load_time_ms"),
            prompt_ms=_metric_avg(metrics_root, "prompt_eval_time_ms"),
            eval_ms=_metric_avg(metrics_root, "eval_time_ms"),
            throughput_eval_tps=_metric_avg(metrics_root, "eval_tokens_per_sec"),
            eval_tokens_avg=_metric_avg(metrics_root, "eval_tokens"),
            memory_mb=_metric_avg(metrics_root, "memory_usage_mb"),
            semantic_similarity=_safe_float(summary_payload, "semantic_similarity_avg"),
            f1=_safe_float(evaluation_block, "f1"),
            retrieval_carbon_avg=retrieval_carbon_avg,
            generation_carbon_avg=generation_carbon_avg,
            question_count=question_count,
        )
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_carbon_split_chart(reports: List[ModelReport]) -> go.Figure:
    labels = [r.label for r in reports]
    retrieval = [r.retrieval_carbon_avg for r in reports]
    generation = [r.generation_carbon_avg for r in reports]
    fig = go.Figure()
    fig.add_bar(name="Retrieval", x=labels, y=retrieval, marker_color="#06b6d4")
    fig.add_bar(name="Generation", x=labels, y=generation, marker_color="#f97316")
    fig.update_layout(
        title="Per-Question Carbon Split",
        xaxis_title="Model",
        yaxis_title="kg CO₂e",
        barmode="stack",
        template="plotly_white",
    )
    return fig


def build_accuracy_vs_carbon_chart(reports: List[ModelReport]) -> go.Figure:
    labels: List[str] = []
    x_vals: List[float] = []
    y_vals: List[float] = []
    for report in reports:
        if report.blended_accuracy is None or report.generation_carbon_avg is None:
            continue
        labels.append(report.label)
        x_vals.append(report.generation_carbon_avg)
        y_vals.append(report.blended_accuracy)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=y_vals,
                text=labels,
                mode="markers+text",
                textposition="top center",
                marker=dict(size=14, color="#2563eb", line=dict(width=1, color="#1e1b4b")),
            )
        ]
    )
    fig.update_layout(
        title="Blended Accuracy vs Generation Carbon",
        xaxis_title="Generation Carbon per Question (kg)",
        yaxis_title="(F1 + Semantic) / 2",
        template="plotly_white",
    )
    return fig


def build_latency_vs_carbon_bubble(reports: List[ModelReport]) -> go.Figure:
    x_vals: List[float] = []
    y_vals: List[float] = []
    sizes: List[float] = []
    labels: List[str] = []
    for report in reports:
        if report.generation_latency_ms is None or report.generation_carbon_avg is None or report.retrieval_latency_ms is None:
            continue
        labels.append(report.label)
        x_vals.append(report.generation_latency_ms)
        y_vals.append(report.generation_carbon_avg)
        sizes.append(max(report.retrieval_latency_ms, 0.1) * 80.0)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=sizes,
                    sizemode="diameter",
                    sizemin=12,
                    color="#22c55e",
                    opacity=0.7,
                ),
            )
        ]
    )
    fig.update_layout(
        title="Generation Latency vs Carbon (bubble size = retrieval latency)",
        xaxis_title="Generation Latency (ms)",
        yaxis_title="Generation Carbon per Question (kg)",
        template="plotly_white",
    )
    return fig


def build_retrieval_latency_vs_carbon(reports: List[ModelReport]) -> go.Figure:
    labels = []
    latencies = []
    carbons = []
    for report in reports:
        if report.retrieval_latency_ms is None or report.retrieval_carbon_avg is None:
            continue
        labels.append(report.label)
        latencies.append(report.retrieval_latency_ms)
        carbons.append(report.retrieval_carbon_avg)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=latencies,
                y=carbons,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=12, color="#0ea5e9"),
            )
        ]
    )
    fig.update_layout(
        title="Retrieval Latency vs Retrieval Carbon",
        xaxis_title="Retrieval Latency (ms)",
        yaxis_title="Retrieval Carbon per Question (kg)",
        template="plotly_white",
    )
    return fig


def build_throughput_vs_carbon(reports: List[ModelReport]) -> go.Figure:
    labels = []
    throughput = []
    carbon_intensity = []
    for report in reports:
        intensity = report.generation_carbon_per_100_tokens
        if report.throughput_eval_tps is None or intensity is None:
            continue
        labels.append(report.label)
        throughput.append(report.throughput_eval_tps)
        carbon_intensity.append(intensity)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=throughput,
                y=carbon_intensity,
                mode="markers+text",
                text=labels,
                marker=dict(size=14, color="#a855f7"),
            )
        ]
    )
    fig.update_layout(
        title="Eval Throughput vs Carbon Intensity",
        xaxis_title="Eval Throughput (tokens/sec)",
        yaxis_title="Generation Carbon per 100 Tokens (kg)",
        template="plotly_white",
    )
    return fig


def build_carbon_ratio_chart(reports: List[ModelReport]) -> go.Figure:
    labels = []
    ratios = []
    for report in reports:
        ratio = report.carbon_ratio
        if ratio is None:
            continue
        labels.append(report.label)
        ratios.append(ratio)
    fig = go.Figure(
        data=[go.Bar(x=labels, y=ratios, marker_color="#facc15")]
    )
    fig.update_layout(
        title="Retrieval-to-Generation Carbon Ratio",
        xaxis_title="Model",
        yaxis_title="Retrieval Carbon / Generation Carbon",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def build_insights(reports: List[ModelReport]) -> str:
    bullets: List[str] = []
    best_efficiency = _best_by(reports, key=lambda r: r.generation_carbon_per_100_tokens, reverse=False)
    if best_efficiency:
        bullets.append(
            f"<li><strong>{best_efficiency.label}</strong> has the lowest generation carbon intensity ({best_efficiency.generation_carbon_per_100_tokens:.2e} kg/100 tok).</li>"
        )
    best_accuracy = _best_by(reports, key=lambda r: r.blended_accuracy)
    if best_accuracy:
        bullets.append(
            f"<li><strong>{best_accuracy.label}</strong> leads on blended accuracy ({best_accuracy.blended_accuracy:.3f}).</li>"
        )
    lowest_latency = _best_by(reports, key=lambda r: r.total_latency_ms, reverse=False)
    if lowest_latency:
        bullets.append(
            f"<li><strong>{lowest_latency.label}</strong> is fastest end-to-end ({lowest_latency.total_latency_ms:.1f} ms).</li>"
        )
    best_ratio = _best_by(reports, key=lambda r: r.carbon_ratio, reverse=False)
    if best_ratio:
        bullets.append(
            f"<li><strong>{best_ratio.label}</strong> keeps retrieval carbon proportion lowest (ratio {best_ratio.carbon_ratio:.2f}).</li>"
        )
    if not bullets:
        bullets.append("<li>No insights available (missing metrics).</li>")
    return "<ul>" + "".join(bullets) + "</ul>"


def _best_by(reports: Iterable[ModelReport], key, reverse: bool = True) -> Optional[ModelReport]:
    scored: List[Tuple[float, ModelReport]] = []
    for report in reports:
        value = key(report)
        if isinstance(value, (int, float)):
            scored.append((float(value), report))
    if not scored:
        return None
    value, report = max(scored, key=lambda pair: pair[0]) if reverse else min(scored, key=lambda pair: pair[0])
    return report


def render_dashboard(reports: List[ModelReport], output: Path) -> None:
    if not reports:
        raise SystemExit("No ragres2 summaries found.")

    charts = [
        (build_carbon_split_chart(reports), "carbon_split"),
        (build_accuracy_vs_carbon_chart(reports), "accuracy_vs_carbon"),
        (build_latency_vs_carbon_bubble(reports), "latency_vs_carbon"),
        (build_retrieval_latency_vs_carbon(reports), "retrieval_vs_carbon"),
        (build_throughput_vs_carbon(reports), "throughput_vs_carbon"),
        (build_carbon_ratio_chart(reports), "carbon_ratio"),
    ]

    chart_html_blocks = []
    for fig, div_id in charts:
        chart_html_blocks.append(
            pio.to_html(fig, include_plotlyjs="cdn" if not chart_html_blocks else False, full_html=False, div_id=div_id)
        )

    insights_html = build_insights(reports)

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>RAG Benchmark Dashboard (ragres2)</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; padding: 2rem; background: #f8fafc; }}
h1 {{ margin-top: 0; }}
section {{ margin-bottom: 2rem; background: #fff; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08); }}
ul {{ margin: 0; padding-left: 1.25rem; }}
</style>
</head>
<body>
<h1>RAG Benchmark Overview (ragres2)</h1>
<section>
<h2>Quick Insights</h2>
{insights_html}
</section>
<section>
<h2>Comparisons</h2>
{"".join(chart_html_blocks)}
</section>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.rag_root).resolve()
    reports = collect_reports(root)
    render_dashboard(reports, Path(args.output).resolve())


if __name__ == "__main__":
    main()
