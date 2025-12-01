#!/usr/bin/env python3
"""Generate a combined two-tab dashboard for QnA and Summarization benchmarks."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Tuple

import plotly.graph_objects as go


def load_dashboard_module(module_path: Path, module_name: str, *, reset_baseline: bool = False) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Dashboard module not found: {module_path}")
    if reset_baseline and "llm_benchmark_dashboard" in sys.modules:
        sys.modules.pop("llm_benchmark_dashboard")
    if module_name in sys.modules:
        del sys.modules[module_name]
    module_dir = str(module_path.parent)
    sys.path.insert(0, module_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[call-arg]
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module
    finally:
        sys.path.remove(module_dir)


def build_bundles(module: ModuleType, root: Path) -> Tuple[List[Tuple[str, str, go.Figure]], str]:
    runs = module.baseline.collect_model_runs(root)  # type: ignore[attr-defined]
    if not runs:
        message = f"No model runs found in {root}"
        return [], message
    bundles = list(module.baseline.build_graphs(runs))  # type: ignore[attr-defined]
    emission_bundle = module.build_carbon_emission_bundle(root)  # type: ignore[attr-defined]
    if emission_bundle:
        bundles.append(emission_bundle)
    html = module.render_section_html(bundles)  # type: ignore[attr-defined]
    return bundles, html


def build_empty_section(title: str, message: str) -> str:
    return (
        f"<section class=\"card\"><h2>{title}</h2>"
        f"<p class=\"description\">{message}</p></section>"
    )


def compose_html(qna_html: str, summarization_html: str) -> str:
    head = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<title>LLM Benchmarks – QnA & Summarization</title>
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
.card.card-wide { grid-column: 1 / -1; }
.description { color: #374151; margin-top: 0; }
.card .plotly-graph-div { width: 100% !important; }
</style>
</head>
<body>
<h1>LLM Benchmarks</h1>
<div class='tabs'>
<button class='tab-button active' data-target='qna'>QnA</button>
<button class='tab-button' data-target='summarization'>Summarization</button>
</div>
<div id='qna' class='tab-content active'><div class='grid'>"""
    tail = """</div></div>
<div id='summarization' class='tab-content'><div class='grid'>""" + summarization_html + """</div></div>
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
    return head + qna_html + tail


def generate_dashboard(qna_root: Path, summarization_root: Path, destination: Path) -> None:
    qna_module_path = qna_root / "llm_dashboard_with_live.py"
    summarization_module_path = summarization_root / "llm_dashboard_with_live.py"

    qna_module = load_dashboard_module(qna_module_path, "qna_dashboard")
    qna_bundles, qna_html = build_bundles(qna_module, qna_root)
    if not qna_bundles:
        qna_html = build_empty_section("QnA Benchmarks", qna_html or "No runs were found.")

    summarization_module = load_dashboard_module(
        summarization_module_path,
        "summarization_dashboard",
        reset_baseline=True,
    )
    summarization_bundles, summarization_html = build_bundles(summarization_module, summarization_root)
    if not summarization_bundles:
        summarization_html = build_empty_section("Summarization Benchmarks", summarization_html or "No runs were found.")

    html_output = compose_html(qna_html, summarization_html)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_output, encoding="utf-8")
    print(f"Wrote combined dashboard to {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined QnA + Summarization dashboard")
    parser.add_argument("--qna-root", default="../QnA", help="Path to the QnA model directory root")
    parser.add_argument("--summ-root", default="../Summarization", help="Path to the Summarization model directory root")
    parser.add_argument("--out", default="combined_dashboard.html", help="Destination HTML path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qna_root = Path(args.qna_root).resolve()
    summ_root = Path(args.summ_root).resolve()
    out_path = Path(args.out).resolve()
    generate_dashboard(qna_root, summ_root, out_path)


if __name__ == "__main__":
    main()

