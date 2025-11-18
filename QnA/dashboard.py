#!/usr/bin/env python3
"""Interactive dashboard for comparing benchmarked models.

The script loads ``summary_metrics.json`` and ``evaluation_metrics.json`` from
all model sub-directories under the requested root, builds twelve analytic plots,
and serves them through a Dash app (or exports a standalone HTML file).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from dash import Dash, dcc, html


PAGE_STYLE = {
    "minHeight": "100vh",
    "backgroundColor": "#f6f7fb",
    "padding": "32px 40px",
    "fontFamily": "'Inter', Arial, sans-serif",
}

GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
    "gap": "28px",
}

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "18px",
    "padding": "18px 24px",
    "boxShadow": "0 18px 35px rgba(15, 23, 42, 0.08)",
    "border": "1px solid #e5e7eb",
}

DESCRIPTION_STYLE = {
    "color": "#4b5563",
    "marginTop": "4px",
    "marginBottom": "18px",
}


@dataclass
class ModelRun:
    name: str
    summary: Dict[str, Any]
    evaluation: Dict[str, Any]

    @property
    def per_question_summary(self) -> List[Dict[str, Any]]:
        return list(self.summary.get("per_question") or [])

    @property
    def per_question_eval(self) -> List[Dict[str, Any]]:
        return list(self.evaluation.get("per_question") or [])


@dataclass
class FigureBundle:
    title: str
    description: str
    figure: go.Figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize benchmark metrics across models")
    parser.add_argument(
        "--root",
        default=".",
        help="Directory that contains the individual model folders",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Dash server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Dash server (default: 8050)",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Skip launching the Dash server (useful when only exporting HTML)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Optional path to write a standalone HTML dashboard",
    )
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_model_runs(root: Path) -> List[ModelRun]:
    runs: List[ModelRun] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        summary_path = candidate / "summary_metrics.json"
        evaluation_path = candidate / "evaluation_metrics.json"
        summary = load_json(summary_path)
        evaluation = load_json(evaluation_path)
        if summary and evaluation:
            runs.append(ModelRun(candidate.name, summary, evaluation))
    return runs


def get_metric(summary: Dict[str, Any], metric_name: str, stat: str = "avg") -> Optional[float]:
    metrics = summary.get("metrics", {})
    target = metrics.get(metric_name)
    if isinstance(target, dict):
        value = target.get(stat)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def fetch_overall_metric(evaluation: Dict[str, Any], metric_name: str) -> Optional[float]:
    overall = evaluation.get("overall", {})
    value = overall.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_question_labels(runs: List[ModelRun]) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for run in runs:
        for entry in run.per_question_eval:
            qid = entry.get("id")
            if isinstance(qid, int) and qid not in labels:
                labels[qid] = entry.get("question", f"Question {qid}")
    return labels


def build_latency_vs_accuracy(runs: List[ModelRun]) -> go.Figure:
    xs: List[float] = []
    ys: List[float] = []
    texts: List[str] = []
    for run in runs:
        latency = get_metric(run.summary, "total_time_ms")
        accuracy = fetch_overall_metric(run.evaluation, "f1")
        if latency is None or accuracy is None:
            continue
        xs.append(latency)
        ys.append(accuracy)
        texts.append(run.name)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                text=texts,
                marker=dict(size=16, color=ys, colorscale="Viridis", showscale=True),
                hovertemplate="Model: %{text}<br>Latency: %{x:.1f} ms<br>F1: %{y:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Latency vs Accuracy<br><sup>Compare average total runtime (ms) against overall F1 per model.</sup>",
        xaxis_title="Average total time (ms)",
        yaxis_title="Overall F1",
    )
    return fig


def build_latency_breakdown(runs: List[ModelRun]) -> go.Figure:
    model_names = [run.name for run in runs]
    segments = {
        "Load": [get_metric(run.summary, "load_time_ms") for run in runs],
        "Prompt": [get_metric(run.summary, "prompt_eval_time_ms") for run in runs],
        "Generation": [get_metric(run.summary, "eval_time_ms") for run in runs],
        "Sampling": [get_metric(run.summary, "sampling_time_ms") for run in runs],
    }
    bars = [
        go.Bar(name=label, x=model_names, y=values)
        for label, values in segments.items()
    ]
    fig = go.Figure(data=bars)
    fig.update_layout(
        barmode="stack",
        title="Latency Breakdown<br><sup>Stacked average stage durations expose where each model spends time.</sup>",
        yaxis_title="Milliseconds",
    )
    return fig


def build_throughput_line(runs: List[ModelRun]) -> go.Figure:
    model_names = [run.name for run in runs]
    throughputs = [get_metric(run.summary, "throughput_tokens_per_sec") for run in runs]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=model_names,
                y=throughputs,
                mode="lines+markers",
            )
        ]
    )
    fig.update_layout(
        title="Tokens per Second<br><sup>Average end-to-end throughput for each model.</sup>",
        yaxis_title="Tokens / second",
    )
    return fig


def build_memory_boxplot(runs: List[ModelRun]) -> go.Figure:
    traces: List[go.Box] = []
    for run in runs:
        samples = [entry.get("memory_avg_mb") for entry in run.per_question_summary if isinstance(entry.get("memory_avg_mb"), (int, float))]
        if not samples:
            continue
        traces.append(
            go.Box(
                y=samples,
                name=run.name,
                boxmean="sd",
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Memory Envelope<br><sup>Distribution of observed average memory usage per question (MB).</sup>",
        yaxis_title="Average memory (MB)",
    )
    return fig


def build_accuracy_violin(runs: List[ModelRun]) -> go.Figure:
    traces: List[go.Violin] = []
    for run in runs:
        scores = [entry.get("f1") for entry in run.per_question_summary if isinstance(entry.get("f1"), (int, float))]
        if not scores:
            continue
        traces.append(
            go.Violin(
                y=scores,
                name=run.name,
                spanmode="hard",
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Accuracy Distribution<br><sup>Per-question F1 spread indicates consistency.</sup>",
        yaxis_title="F1 score",
    )
    return fig


def build_accuracy_heatmap(runs: List[ModelRun], question_labels: Dict[int, str]) -> go.Figure:
    model_names = [run.name for run in runs]
    question_ids = sorted({entry.get("id") for run in runs for entry in run.per_question_summary if isinstance(entry.get("id"), int)})
    z_values: List[List[Optional[float]]] = []
    y_labels: List[str] = []
    custom_data: List[List[str]] = []
    for qid in question_ids:
        row: List[Optional[float]] = []
        custom_row: List[str] = []
        for run in runs:
            match = next((entry for entry in run.per_question_summary if entry.get("id") == qid), None)
            value = match.get("exact_match") if match else None
            row.append(value)
            question_text = question_labels.get(qid, f"Question {qid}")
            custom_row.append(question_text)
        y_labels.append(f"Q{qid:02d}")
        z_values.append(row)
        custom_data.append(custom_row)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=model_names,
                y=y_labels,
                zmin=0,
                zmax=1,
                colorscale=[[0, "#f94144"], [1, "#577590"]],
                colorbar=dict(title="Exact match"),
                customdata=custom_data,
                hovertemplate="Model: %{x}<br>%{y} · %{customdata}<br>Exact match: %{z}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Question-Level Exact Match<br><sup>Binary correctness per question highlights strengths and gaps.</sup>",
        xaxis_title="Model",
        yaxis_title="Questions",
    )
    return fig


def build_throughput_vs_memory(runs: List[ModelRun]) -> go.Figure:
    xs: List[float] = []
    ys: List[float] = []
    texts: List[str] = []
    for run in runs:
        throughput = get_metric(run.summary, "throughput_tokens_per_sec")
        memory = get_metric(run.summary, "memory_avg_mb")
        if throughput is None or memory is None:
            continue
        xs.append(throughput)
        ys.append(memory)
        texts.append(run.name)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=texts,
                textposition="top center",
                marker=dict(size=14, color=xs, colorscale="Plasma", showscale=True),
                hovertemplate="Model: %{text}<br>Throughput: %{x:.2f} tok/s<br>Avg memory: %{y:.1f} MB<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Throughput vs Memory Footprint<br><sup>Identify the fastest model that still fits your device.</sup>",
        xaxis_title="Tokens per second",
        yaxis_title="Average memory (MB)",
    )
    return fig


def build_graphs(runs: List[ModelRun]) -> List[FigureBundle]:
    question_labels = build_question_labels(runs)
    figures = [
        FigureBundle("Latency vs Accuracy", "Lower latency and higher accuracy is the sweet spot.", build_latency_vs_accuracy(runs)),
        FigureBundle("Latency Breakdown", "Stacked averages for load, prompt, generation, and sampling stages.", build_latency_breakdown(runs)),
        FigureBundle("Tokens per Second", "Overall throughput captured directly from llama.cpp metrics.", build_throughput_line(runs)),
        FigureBundle("Memory Envelope", "Distribution of average memory consumption across questions.", build_memory_boxplot(runs)),
        FigureBundle("Accuracy Distribution", "Violin plots of per-question F1 scores.", build_accuracy_violin(runs)),
        FigureBundle("Exact Match Heatmap", "Question-by-question EM to spot topical weaknesses.", build_accuracy_heatmap(runs, question_labels)),
        FigureBundle("Throughput vs Memory", "Choose the fastest model that still fits your device.", build_throughput_vs_memory(runs)),
    ]
    return figures


def export_dashboard(bundles: List[FigureBundle], destination: Path) -> None:
    snippets = []
    for bundle in bundles:
        figure_html = bundle.figure.to_html(full_html=False, include_plotlyjs=False)
        snippets.append(
            f"""
            <section class=\"card\">
              <h2>{bundle.title}</h2>
              <p class=\"description\">{bundle.description}</p>
              {figure_html}
            </section>
            """
        )
        html_output = f"""<!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"utf-8\" />
            <title>Benchmark Dashboard</title>
            <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
            <style>
                :root {{ font-family: 'Inter', Arial, sans-serif; color: #0f172a; }}
                body {{ margin: 0; background: #f6f7fb; }}
                main {{ min-height: 100vh; padding: 32px 40px; }}
                h1 {{ text-align: center; margin-top: 0; margin-bottom: 32px; font-weight: 600; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 28px; }}
                .card {{ background: #fff; border-radius: 18px; padding: 18px 24px; box-shadow: 0 18px 35px rgba(15,23,42,.08); border: 1px solid #e5e7eb; }}
                .card h2 {{ margin-bottom: 4px; font-size: 1.15rem; }}
                .description {{ color: #4b5563; margin-top: 0; margin-bottom: 18px; line-height: 1.4; }}
            </style>
        </head>
        <body>
            <main>
                <h1>Benchmark Comparison Dashboard</h1>
                <div class=\"grid\">
                    {''.join(snippets)}
                </div>
            </main>
        </body>
        </html>
        """
    destination.write_text(html_output, encoding="utf-8")
    print(f"Exported dashboard to {destination}")


def launch_dash(bundles: List[FigureBundle], host: str, port: int) -> None:
    app = Dash(
        __name__,
        external_stylesheets=[
            "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css"
        ],
    )
    app.layout = html.Div(
        [
            html.H1(
                "Benchmark Comparison Dashboard",
                style={"textAlign": "center", "marginTop": 0, "marginBottom": "32px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(bundle.title, style={"marginBottom": "4px"}),
                            html.P(bundle.description, style=DESCRIPTION_STYLE),
                            dcc.Graph(figure=bundle.figure),
                        ],
                        style=CARD_STYLE,
                    )
                    for bundle in bundles
                ],
                style=GRID_STYLE,
            ),
        ],
        style=PAGE_STYLE,
    )
    app.run_server(host=host, port=port)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    runs = collect_model_runs(root)
    if not runs:
        raise SystemExit(f"No model runs with summary/evaluation metrics were found in {root}")

    figures = build_graphs(runs)

    if args.export:
        export_dashboard(figures, Path(args.export).resolve())

    if not args.no_serve:
        launch_dash(figures, args.host, args.port)


if __name__ == "__main__":
    main()
