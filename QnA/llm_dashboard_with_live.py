#!/usr/bin/env python3
"""Export a combined dashboard with benchmark plots and live metrics plots.

This leverages the existing figures from ``llm_benchmark_dashboard.py`` for the
benchmark tab and adds a second tab that visualises the ``metrics.csv`` files
present under the ``Live_*`` folders inside each model directory.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import plotly.graph_objects as go

import llm_benchmark_dashboard as baseline


EXCLUDED_METRIC_NAMES = {
    "gpu memory bus busy",
    "gpu utilization",
    "used memory",
}


@dataclass
class Series:
    model: str
    run_label: str
    timestamps: List[int]
    values: List[float]
    unit: str

    def as_relative_seconds(self) -> Tuple[List[float], List[float]]:
        if not self.timestamps:
            return [], []
        base = min(self.timestamps)
        xs = [(ts - base) / 1_000_000.0 for ts in self.timestamps]
        ys = list(self.values)
        return xs, ys


def discover_live_dirs(model_dir: Path) -> Optional[Path]:
    candidates = [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("Live")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def downsample(xs: List[float], ys: List[float], max_points: int = 2000) -> Tuple[List[float], List[float]]:
    if len(xs) <= max_points:
        return xs, ys
    step = max(1, len(xs) // max_points)
    return xs[::step], ys[::step]


def read_metrics_csv(path: Path) -> Dict[Tuple[str, str], Series]:
    series_map: Dict[Tuple[str, str], Series] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        for row in reader:
            capability = (row.get("capabilityName") or "Unknown").strip() or "Unknown"
            name = (row.get("name") or "Unnamed Metric").strip() or "Unnamed Metric"
            subgroup = (row.get("subGroup") or "").strip()
            submetric = (row.get("subMetricName") or "").strip()
            if subgroup or submetric:
                continue  # Skip per-process or sub-metric rows for now.
            try:
                timestamp = int(row.get("timestamp") or 0)
                value = float(row.get("value") or 0.0)
            except ValueError:
                continue
            unit = (row.get("unit") or "").strip()
            key = (capability, name)
            series = series_map.setdefault(key, Series("", path.parent.name, [], [], unit))
            series.timestamps.append(timestamp)
            series.values.append(value)
    return series_map


def collect_live_series(root: Path) -> Dict[Tuple[str, str], List[Series]]:
    aggregated: Dict[Tuple[str, str], List[Series]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        latest_live = discover_live_dirs(model_dir)
        if latest_live is None:
            continue
        metrics_csv = latest_live / "metrics.csv"
        if not metrics_csv.exists():
            continue
        series_map = read_metrics_csv(metrics_csv)
        for key, series in series_map.items():
            series.model = model_dir.name
            aggregated.setdefault(key, []).append(series)
    return aggregated


def build_live_figures(series_by_metric: Dict[Tuple[str, str], List[Series]]) -> List[Tuple[str, str, go.Figure]]:
    bundles: List[Tuple[str, str, go.Figure]] = []
    for (capability, metric_name), series_list in sorted(series_by_metric.items()):
        if not series_list:
            continue
        if metric_name.strip().lower() in EXCLUDED_METRIC_NAMES:
            continue
        traces = []
        unit = series_list[0].unit
        for series in series_list:
            xs, ys = series.as_relative_seconds()
            if not xs:
                continue
            xs, ys = downsample(xs, ys)
            label = f"{series.model} ({series.run_label})"
            traces.append(go.Scatter(x=xs, y=ys, mode="lines", name=label))
        if not traces:
            continue
        fig = go.Figure(data=traces)
        yaxis_title = f"Value ({unit})" if unit else "Value"
        fig.update_layout(
            title=f"{capability} – {metric_name}",
            xaxis_title="Time since start (s)",
            yaxis_title=yaxis_title,
            legend_title="Model (Live run)",
        )
        description = (
            f"Latest Live capture for each model. Shows {metric_name} under the {capability} capability "
            "over time using the aggregate rows from metrics.csv."
        )
        bundles.append((f"{capability}: {metric_name}", description, fig))
    return bundles


def render_section_html(bundles: Iterable[Tuple[str, str, go.Figure]]) -> str:
    snippets: List[str] = []
    for title, desc, fig in bundles:
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
        snippets.append(
            f"<section class=\"card\"><h2>{title}</h2><p class=\"description\">{desc}</p>{chart_html}</section>"
        )
    return "".join(snippets)


def export_combined_dashboard(benchmark_bundles, live_bundles, destination: Path) -> None:
    benchmark_html = render_section_html(benchmark_bundles)
    if live_bundles:
        live_html = render_section_html(live_bundles)
    else:
        live_html = "<section class=\"card\"><h2>Live Metrics</h2><p class=\"description\">No metrics.csv files were found.</p></section>"
    head = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<title>LLM Benchmark Dashboard</title>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<style>
body { font-family: Inter, Arial, sans-serif; margin: 0; padding: 2rem; background: #f6f8fb; }
h1 { text-align: center; margin-bottom: 1rem; }
.tabs { display: flex; justify-content: center; gap: 12px; margin-bottom: 1.5rem; }
.tab-button { padding: 0.6rem 1.4rem; border: none; border-radius: 999px; background: #d1d5db; color: #1f2937; font-weight: 600; cursor: pointer; transition: all 0.2s ease; }
.tab-button.active { background: #2563eb; color: white; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }
.card { background: #fff; border-radius: 12px; padding: 14px; box-shadow: 0 6px 20px rgba(10,20,30,0.06); }
.description { color: #374151; margin-top: 0; }
</style>
</head>
<body>
<h1>LLM Benchmark Dashboard</h1>
<div class='tabs'>
<button class='tab-button active' data-target='benchmark'>Benchmark Metrics</button>
<button class='tab-button' data-target='live'>Live Metrics</button>
</div>
<div id='benchmark' class='tab-content active'><div class='grid'>"""
    tail = """</div></div>
<div id='live' class='tab-content'><div class='grid'>""" + live_html + """</div></div>
<script>
for (const btn of document.querySelectorAll('.tab-button')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    btn.classList.add('active');
    const target = document.getElementById(btn.dataset.target);
    if (target) { target.classList.add('active'); }
  });
}
</script>
</body>
</html>"""
    html_output = head + benchmark_html + tail
    destination.write_text(html_output, encoding='utf-8')
    print(f"Wrote dashboard to {destination}")


def main(root: Path, out_path: Path) -> None:
    runs = baseline.collect_model_runs(root)
    if not runs:
        raise SystemExit(f"No model runs found in {root}")
    benchmark_bundles = baseline.build_graphs(runs)
    live_bundles = build_live_figures(collect_live_series(root))
    export_combined_dashboard(benchmark_bundles, live_bundles, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate combined benchmark/live dashboard')
    parser.add_argument('--root', default='.', help='Root directory containing model folders')
    parser.add_argument('--out', default='dashboard_with_live.html', help='Destination HTML path')
    args = parser.parse_args()
    main(Path(args.root).resolve(), Path(args.out).resolve())
