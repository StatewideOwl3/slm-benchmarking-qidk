#!/usr/bin/env python3
"""Generate composite efficiency dashboards for RAG, QnA, and Summarization tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


@dataclass
class ModelPoint:
    label: str
    quality: float  # (F1 + semantic) / 2 when available
    latency_s: float  # prompt + eval latency in seconds
    carbon_g: float  # per-question carbon in grams (retrieval + generation when applicable)
    question_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composite efficiency page builder")
    parser.add_argument("--project-root", default=".", help="Path to repo root (default: current directory)")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where HTML pages will be written (default: current directory)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_avg(payload: Dict[str, object], key: str) -> Optional[float]:
    block = payload.get("metrics", {}).get(key)
    if isinstance(block, dict):
        avg = block.get("avg")
        if isinstance(avg, (int, float)):
            return float(avg)
    return None


def _find_parsed_file(model_dir: Path) -> Optional[Path]:
    matches = sorted(model_dir.glob("parsed_outputs_*.json"))
    return matches[0] if matches else None


def _collect_rag_models(root: Path) -> List[ModelPoint]:
    reports: List[ModelPoint] = []
    for summary_path in sorted((root / "RAG").glob("*ragres2*/*/summary.json")):
        model_dir = summary_path.parent
        summary_metrics_path = model_dir / "summary_metrics.json"
        parsed_path = _find_parsed_file(model_dir)
        if not summary_metrics_path.exists() or parsed_path is None:
            continue

        summary_payload = _load_json(summary_path)
        sm_payload = _load_json(summary_metrics_path)
        parsed_entries = _load_json(parsed_path)
        if not isinstance(parsed_entries, list) or not parsed_entries:
            continue

        semantic = summary_payload.get("semantic_similarity_avg")
        evaluation = summary_payload.get("evaluation") or {}
        f1 = evaluation.get("f1")
        if not isinstance(f1, (int, float)):
            continue
        quality = float(f1)
        if isinstance(semantic, (int, float)):
            quality = 0.5 * (float(f1) + float(semantic))

        prompt_ms = _metric_avg(sm_payload, "prompt_eval_time_ms")
        eval_ms = _metric_avg(sm_payload, "eval_time_ms")
        if prompt_ms is None or eval_ms is None:
            continue
        latency_s = (prompt_ms + eval_ms) / 1000.0

        retrieval_carbon: List[float] = []
        generation_carbon: List[float] = []
        for entry in parsed_entries:
            rc = entry.get("retrieval_carbon_kg")
            gc = entry.get("generation_carbon_kg") or entry.get("carbon_emissions_kg")
            if isinstance(rc, (int, float)):
                retrieval_carbon.append(float(rc))
            if isinstance(gc, (int, float)):
                generation_carbon.append(float(gc))
        if not generation_carbon:
            continue
        retr_avg = mean(retrieval_carbon) if retrieval_carbon else 0.0
        gen_avg = mean(generation_carbon)
        carbon_g = (retr_avg + gen_avg) * 1000.0  # kg -> g

        question_count = int(summary_payload.get("question_count") or len(parsed_entries))
        reports.append(
            ModelPoint(
                label=model_dir.parent.name,
                quality=quality,
                latency_s=latency_s,
                carbon_g=carbon_g,
                question_count=question_count,
            )
        )
    return reports


def _collect_task_models(task_root: Path, question_key: str) -> List[ModelPoint]:
    reports: List[ModelPoint] = []
    for model_dir in sorted(task_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        eval_path = model_dir / "evaluation_metrics.json"
        sm_path = model_dir / "summary_metrics.json"
        if not eval_path.exists() or not sm_path.exists():
            continue

        evaluation = _load_json(eval_path).get("overall", {})
        f1 = evaluation.get("f1")
        if not isinstance(f1, (int, float)):
            continue
        quality = float(f1)  # semantic similarity not logged for these tasks

        carbon_kg = evaluation.get("carbon_emissions_kg_avg")
        if not isinstance(carbon_kg, (int, float)):
            continue
        carbon_g = float(carbon_kg) * 1000.0

        sm_payload = _load_json(sm_path)
        prompt_ms = _metric_avg(sm_payload, "prompt_eval_time_ms")
        eval_ms = _metric_avg(sm_payload, "eval_time_ms")
        if prompt_ms is None or eval_ms is None:
            continue
        latency_s = (prompt_ms + eval_ms) / 1000.0

        question_count_value = evaluation.get(question_key) or sm_payload.get("question_count")
        question_count = int(question_count_value) if question_count_value else 0

        reports.append(
            ModelPoint(
                label=model_dir.name,
                quality=quality,
                latency_s=latency_s,
                carbon_g=carbon_g,
                question_count=question_count,
            )
        )
    return reports


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _normalize_models(models: List[ModelPoint]) -> List[Dict[str, float]]:
        if not models:
                return []
        max_latency = max(m.latency_s for m in models) or 1.0
        max_carbon = max(m.carbon_g for m in models) or 1.0
        dataset = []
        for m in models:
                dataset.append(
                        {
                                "model": m.label,
                                "quality": m.quality,
                                "latency": m.latency_s,
                                "latencyNorm": m.latency_s / max_latency,
                                "carbon": m.carbon_g,
                                "carbonNorm": m.carbon_g / max_carbon,
                                "questionCount": m.question_count,
                        }
                )
        return dataset


def _render_merged_page(
        rag_models: List[ModelPoint],
        qna_models: List[ModelPoint],
        sum_models: List[ModelPoint],
        output: Path,
) -> None:
        if not (rag_models and qna_models and sum_models):
                raise SystemExit("Expected models for all three tasks (RAG, QnA, Summarization)")

        payload = json.dumps(
                {
                        "RAG": _normalize_models(rag_models),
                        "QnA": _normalize_models(qna_models),
                        "Summarization": _normalize_models(sum_models),
                },
                ensure_ascii=False,
        )

        html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Composite Efficiency Dashboard</title>
<script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
<style>
body {{ font-family: 'Inter', Arial, sans-serif; margin: 0; padding: 2rem; background: #f8fafc; color: #0f172a; }}
h1 {{ margin-bottom: 0.5rem; }}
.tabs {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
.tab {{ cursor: pointer; padding: 0.75rem 1.25rem; border-radius: 999px; background: #e2e8f0; font-weight: 600; }}
.tab.active {{ background: #2563eb; color: #fff; }}
.control-panel {{ display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
.control {{ background: #fff; padding: 1rem; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15,23,42,0.08); flex: 1; min-width: 250px; }}
label {{ font-weight: 600; display: block; margin-bottom: 0.25rem; }}
input[type=range] {{ width: 100%; }}
#chart {{ background: #fff; padding: 1rem; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15,23,42,0.08); }}
.note {{ margin-top: 1rem; font-size: 0.95rem; color: #334155; }}
</style>
</head>
<body>
<h1>Composite Efficiency Dashboard</h1>
<p class=\"note\">EffAcc(α, β) = Q / (1 + α·T<sub>norm</sub> + β·C<sub>norm</sub>). Adjust the sliders to emphasize latency or carbon per task.</p>
<div class=\"tabs\">
    <div class=\"tab active\" data-tab=\"RAG\">RAG</div>
    <div class=\"tab\" data-tab=\"QnA\">QnA</div>
    <div class=\"tab\" data-tab=\"Summarization\">Summarization</div>
</div>
<div class=\"control-panel\">
    <div class=\"control\">
        <label for=\"alpha\">Latency Weight (α)</label>
        <input type=\"range\" id=\"alpha\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.20\" />
        <div>α = <span id=\"alphaValue\">0.20</span></div>
    </div>
    <div class=\"control\">
        <label for=\"beta\">Carbon Weight (β)</label>
        <input type=\"range\" id=\"beta\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.20\" />
        <div>β = <span id=\"betaValue\">0.20</span></div>
    </div>
</div>
<div id=\"chart\"></div>
<script>
const DATA = {payload};
let currentTab = 'RAG';
const alphaInput = document.getElementById('alpha');
const betaInput = document.getElementById('beta');
const alphaValue = document.getElementById('alphaValue');
const betaValue = document.getElementById('betaValue');
const chartDiv = document.getElementById('chart');

function computeScores(alpha, beta, dataset) {{
    return dataset.map((item) => {{
        const denominator = 1 + alpha * item.latencyNorm + beta * item.carbonNorm;
        const score = item.quality / denominator;
        return {{ ...item, score }};
    }});
}}

function render() {{
    const alpha = parseFloat(alphaInput.value);
    const beta = parseFloat(betaInput.value);
    const dataset = DATA[currentTab] || [];
    const scores = computeScores(alpha, beta, dataset);
    const trace = {{
        type: 'bar',
        x: scores.map((row) => row.model),
        y: scores.map((row) => row.score),
        text: scores.map((row) => `Q=${'{'}row.quality.toFixed(3){'}'} | T=${'{'}row.latency.toFixed(2){'}'}s | C=${'{'}row.carbon.toExponential(2){'}'} g`),
        textposition: 'outside',
        marker: {{ color: '#2563eb' }},
    }};
    const layout = {{
        title: `${'{'}currentTab{'}'} EffAcc(α, β) per Model`,
        yaxis: {{ title: 'Composite Efficiency' }},
        margin: {{ t: 60, r: 20, b: 80, l: 60 }},
    }};
    Plotly.react(chartDiv, [trace], layout);
}}

function updateSliders() {{
    alphaValue.textContent = parseFloat(alphaInput.value).toFixed(2);
    betaValue.textContent = parseFloat(betaInput.value).toFixed(2);
    render();
}}

alphaInput.addEventListener('input', updateSliders);
betaInput.addEventListener('input', updateSliders);

document.querySelectorAll('.tab').forEach((tab) => {{
    tab.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach((btn) => btn.classList.remove('active'));
        tab.classList.add('active');
        currentTab = tab.dataset.tab;
        render();
    }});
}});

updateSliders();
</script>
</body>
</html>
"""
        output.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rag_models = _collect_rag_models(root)
    qna_models = _collect_task_models(root / "QnA", "question_count")
    sum_models = _collect_task_models(root / "Summarization", "sample_count")

    _render_merged_page(rag_models, qna_models, sum_models, output_dir / "composite_dashboard.html")


if __name__ == "__main__":
    main()
