#!/usr/bin/env python3
"""Export a summarization benchmark dashboard with optional carbon metrics overlay."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import llm_benchmark_dashboard as baseline


KG_TO_G = 1_000.0
KG_TO_MG = 1_000_000.0
COUNT_KEYS = ("question_count", "sample_count", "entry_count")


@dataclass
class CarbonStats:
    model: str
    total_kg: Optional[float]
    avg_kg: Optional[float]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def extract_count(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    for key in COUNT_KEYS:
        if key in payload:
            count = safe_int(payload.get(key))
            if count is not None:
                return count
    return None


def collect_carbon_metrics(root: Path) -> List[CarbonStats]:
    stats: List[CarbonStats] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        eval_path = model_dir / "evaluation_metrics.json"
        summary_path = model_dir / "summary_metrics.json"
        total_kg: Optional[float] = None
        avg_kg: Optional[float] = None
        example_count: Optional[int] = None

        if eval_path.exists():
            try:
                with eval_path.open("r", encoding="utf-8") as fh:
                    eval_data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                eval_data = None
            if isinstance(eval_data, dict):
                overall = eval_data.get("overall", {})
                total_kg = total_kg or safe_float(overall.get("carbon_emissions_kg_sum"))
                avg_kg = avg_kg or safe_float(overall.get("carbon_emissions_kg_avg"))
                example_count = example_count or extract_count(overall) or extract_count(eval_data)

        if (avg_kg is None or total_kg is None or example_count is None) and summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as fh:
                    summary_data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                summary_data = None
            if isinstance(summary_data, dict):
                if example_count is None:
                    example_count = extract_count(summary_data) or example_count
                metrics_block = summary_data.get("metrics", {}).get("carbon_emissions_kg", {})
                if avg_kg is None:
                    avg_kg = safe_float(metrics_block.get("avg"))
                if total_kg is None:
                    per_field = summary_data.get("per_question") or summary_data.get("per_sample") or []
                    per_values = [
                        value
                        for value in (
                            safe_float(record.get("carbon_emissions_kg"))
                            for record in per_field
                        )
                        if value is not None
                    ]
                    if per_values:
                        total_kg = sum(per_values)

        if total_kg is None and avg_kg is not None and example_count is not None:
            total_kg = avg_kg * example_count
        if avg_kg is None and total_kg is not None and example_count is not None and example_count > 0:
            avg_kg = total_kg / example_count

        if avg_kg is None and total_kg is None:
            continue
        stats.append(CarbonStats(model=model_dir.name, total_kg=total_kg, avg_kg=avg_kg))
    return stats


def build_carbon_emission_bundle(root: Path) -> Optional[Tuple[str, str, go.Figure]]:
    stats = collect_carbon_metrics(root)
    if not stats:
        return None

    models = [item.model for item in stats]
    total_in_g = [item.total_kg * KG_TO_G if item.total_kg is not None else None for item in stats]
    avg_in_mg = [item.avg_kg * KG_TO_MG if item.avg_kg is not None else None for item in stats]

    has_totals = any(value is not None for value in total_in_g)
    has_avgs = any(value is not None for value in avg_in_mg)
    if not (has_totals or has_avgs):
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if has_totals:
        fig.add_trace(
            go.Bar(
                x=models,
                y=total_in_g,
                name="Total per run (g)",
                marker_color="#34d399",
                hovertemplate="%{x}<br>Total: %{y:.4f} g<extra></extra>",
            ),
            secondary_y=False,
        )
    if has_avgs:
        fig.add_trace(
            go.Scatter(
                x=models,
                y=avg_in_mg,
                name="Avg per article (mg)",
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=8),
                hovertemplate="%{x}<br>Avg: %{y:.3f} mg<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Carbon Emissions by Model",
        autosize=True,
        height=500,
        margin=dict(t=60, r=40, b=40, l=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified",
        meta={"card_class": "card card-wide"},
    )
    fig.update_xaxes(title_text="Model")
    fig.update_yaxes(title_text="Total emissions per run (g)", secondary_y=False)
    fig.update_yaxes(title_text="Average emissions per article (mg)", secondary_y=True)

    description = (
        "Carbon impact derived from evaluation_metrics.json. Totals are shown in grams "
        "while per-article averages are converted to milligrams so smaller differences remain visible."
    )
    return ("Carbon Emissions", description, fig)


def render_section_html(bundles: Iterable[Tuple[str, str, go.Figure]]) -> str:
    snippets: List[str] = []
    for title, desc, fig in bundles:
        card_class = "card"
        meta = getattr(fig.layout, "meta", None)
        if isinstance(meta, dict):
            card_class = meta.get("card_class", card_class)
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
        snippets.append(
            f"<section class=\"{card_class}\"><h2>{title}</h2><p class=\"description\">{desc}</p>{chart_html}</section>"
        )
    return "".join(snippets)


def export_dashboard(bundles: Iterable[Tuple[str, str, go.Figure]], destination: Path) -> None:
    section_html = render_section_html(bundles)
    head = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<title>Summarization Benchmark Dashboard</title>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<style>
body { font-family: Inter, Arial, sans-serif; margin: 0; padding: 2rem; background: #f6f8fb; }
h1 { text-align: center; margin-bottom: 1rem; }
.subtitle { text-align: center; color: #4b5563; margin-bottom: 2rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }
.card { background: #fff; border-radius: 12px; padding: 14px; box-shadow: 0 6px 20px rgba(10,20,30,0.06); }
.card.card-wide { grid-column: 1 / -1; }
.description { color: #374151; margin-top: 0; }
.card .plotly-graph-div { width: 100% !important; }
</style>
</head>
<body>
<h1>Summarization Benchmark Dashboard</h1>
<p class='subtitle'>Cross-model runtime and quality comparisons plus optional carbon metrics.</p>
<div class='grid'>"""
    tail = """</div>
</body>
</html>"""
    html_output = head + section_html + tail
    destination.write_text(html_output, encoding='utf-8')
    print(f"Wrote dashboard to {destination}")


def main(root: Path, out_path: Path) -> None:
    runs = baseline.collect_model_runs(root)
    if not runs:
        raise SystemExit(f"No model runs found in {root}")
    benchmark_bundles = list(baseline.build_graphs(runs))
    emission_bundle = build_carbon_emission_bundle(root)
    if emission_bundle:
        benchmark_bundles.append(emission_bundle)
    export_dashboard(benchmark_bundles, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate summarization benchmark dashboard')
    parser.add_argument('--root', default='.', help='Root directory containing summarization model folders')
    parser.add_argument('--out', default='summarization_dashboard.html', help='Destination HTML path')
    args = parser.parse_args()
    main(Path(args.root).resolve(), Path(args.out).resolve())

