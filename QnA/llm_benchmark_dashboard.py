#!/usr/bin/env python3
"""
LLM benchmark dashboard exporter.

Reads subfolders under a root directory. Each subfolder must contain:
- summary_metrics.json
- evaluation_metrics.json

Produces a single-file HTML dashboard with interactive Plotly charts.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from statistics import mean
import plotly.graph_objects as go

class ModelRun:
    def __init__(self, name: str, summary: Dict[str, Any], evaluation: Dict[str, Any]):
        self.name = name
        self.summary = summary or {}
        self.evaluation = evaluation or {}

    @property
    def per_question_summary(self) -> List[Dict[str, Any]]:
        return list(self.summary.get('per_question') or [])

def load_json(path: Path):
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)

def collect_model_runs(root: Path) -> List[ModelRun]:
    runs = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith('.'):
            continue
        s = load_json(candidate / 'summary_metrics.json')
        e = load_json(candidate / 'evaluation_metrics.json')
        if s and e:
            runs.append(ModelRun(candidate.name, s, e))
    return runs

def get_metric(summary: Dict[str, Any], metric_name: str, stat: str = 'avg') -> Optional[float]:
    metrics = summary.get('metrics', {})
    target = metrics.get(metric_name)
    if isinstance(target, dict):
        value = target.get(stat)
        if isinstance(value, (int, float)):
            return float(value)
    return None

def fetch_overall_metric(evaluation: Dict[str, Any], metric_name: str) -> Optional[float]:
    overall = evaluation.get('overall', {})
    value = overall.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    return None

def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    return mean(filtered) if filtered else None


def sorted_runs_by_metric(runs: List[ModelRun], metric: str, stat: str = 'avg', reverse: bool = True) -> List[ModelRun]:
    """Sort runs by a metric from summary.metrics or overall evaluation.

    If metric is 'f1' or 'exact_match', prefer the evaluation/overall value.
    """
    def score(run: ModelRun) -> float:
        if metric in ('f1', 'exact_match'):
            v = fetch_overall_metric(run.evaluation, metric)
            return float(v) if isinstance(v, (int, float)) else float('-inf')
        v = get_metric(run.summary, metric, stat=stat)
        return float(v) if isinstance(v, (int, float)) else float('-inf')

    return sorted(runs, key=score, reverse=reverse)


def sort_question_ids_by_f1(runs: List[ModelRun], descending: bool = True) -> List[int]:
    """Return question ids sorted by their mean F1 across runs.

    When descending=True, highest-mean-F1 (easier) questions come first.
    """
    qmap: Dict[int, List[float]] = {}
    for run in runs:
        for e in run.per_question_summary:
            qid = e.get('id')
            f1 = e.get('f1')
            if isinstance(qid, int) and isinstance(f1, (int, float)):
                qmap.setdefault(qid, []).append(float(f1))
    scored = [(qid, (sum(vals) / len(vals))) for qid, vals in qmap.items() if vals]
    scored.sort(key=lambda p: p[1], reverse=descending)
    return [qid for qid, _ in scored]

# --- Figures ---

def build_latency_vs_accuracy(runs: List[ModelRun]) -> go.Figure:
    ordered = sorted_runs_by_metric(runs, 'f1', reverse=True)
    xs, ys, texts = [], [], []
    for run in ordered:
        lat = get_metric(run.summary, 'total_time_ms')
        acc = fetch_overall_metric(run.evaluation, 'f1')
        if lat is None or acc is None:
            continue
        xs.append(lat); ys.append(acc); texts.append(run.name)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode='markers+text', text=texts, textposition='top center',
                               marker=dict(size=14, color=ys, colorscale='Viridis', showscale=True)))
    fig.update_layout(
        title='Latency vs Accuracy',
        xaxis_title='Avg total time (ms)',
        yaxis_title='Overall F1 (0–1)',
        annotations=[dict(text='Each point is a model: x=average total response time (ms), y=overall F1', xref='paper', yref='paper', x=0, y=-0.2, showarrow=False)]
    )
    return fig

def build_latency_breakdown(runs: List[ModelRun]) -> go.Figure:
    ordered = sorted_runs_by_metric(runs, 'total_time_ms', reverse=False)
    model_names = [r.name for r in ordered]
    segments = {
        'Load (ms)': [get_metric(r.summary, 'load_time_ms') for r in ordered],
        'Prompt eval (ms)': [get_metric(r.summary, 'prompt_eval_time_ms') for r in ordered],
        'Generation (ms)': [get_metric(r.summary, 'eval_time_ms') for r in ordered],
        'Sampling (ms)': [get_metric(r.summary, 'sampling_time_ms') for r in ordered],
    }
    bars = [go.Bar(name=k, x=model_names, y=v) for k,v in segments.items()]
    fig = go.Figure(data=bars)
    fig.update_layout(barmode='stack', title='Latency Breakdown (stacked averages by stage)', yaxis_title='Milliseconds (ms)')
    return fig

def build_throughput_line(runs: List[ModelRun]) -> go.Figure:
    ordered = sorted_runs_by_metric(runs, 'throughput_tokens_per_sec', reverse=True)
    model_names = [r.name for r in ordered]
    throughputs = [get_metric(r.summary, 'throughput_tokens_per_sec') for r in ordered]
    fig = go.Figure(go.Scatter(x=model_names, y=throughputs, mode='lines+markers'))
    fig.update_layout(title='Throughput (tokens/sec)', yaxis_title='Tokens / sec')
    return fig

def build_accuracy_bars(runs: List[ModelRun]) -> go.Figure:
    ordered = sorted_runs_by_metric(runs, 'f1', reverse=True)
    model_names = [r.name for r in ordered]
    f1s = [fetch_overall_metric(r.evaluation, 'f1') for r in ordered]
    ems = [fetch_overall_metric(r.evaluation, 'exact_match') for r in ordered]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_names, y=f1s, name='F1'))
    fig.add_trace(go.Bar(x=model_names, y=ems, name='Exact Match'))
    fig.update_layout(barmode='group', title='Overall Accuracy (F1 & EM)', yaxis=dict(range=[0,1], title='Score'))
    return fig

def build_throughput_vs_memory(runs: List[ModelRun]) -> go.Figure:
    ordered = sorted_runs_by_metric(runs, 'throughput_tokens_per_sec', reverse=True)
    xs, ys, texts = [], [], []
    for run in ordered:
        tp = get_metric(run.summary, 'throughput_tokens_per_sec')
        mem = get_metric(run.summary, 'memory_avg_mb')
        if tp is None or mem is None:
            continue
        xs.append(tp); ys.append(mem); texts.append(run.name)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode='markers+text', text=texts, textposition='top center',
                               marker=dict(size=12, color=xs, colorscale='Plasma', showscale=True)))
    fig.update_layout(title='Throughput vs Memory (trade-off)', xaxis_title='Throughput (tokens/sec)', yaxis_title='Avg memory (MB)')
    return fig

def build_latency_cdf(runs: List[ModelRun]) -> go.Figure:
    # Sort runs by average total_time_ms for visual consistency
    ordered = sorted_runs_by_metric(runs, 'total_time_ms', reverse=False)
    traces = []
    for run in ordered:
        samples = sorted(float(entry.get('total_time_ms')) for entry in run.per_question_summary if isinstance(entry.get('total_time_ms'), (int,float)))
        if not samples: continue
        n = len(samples)
        y = [i/(n-1) if n>1 else 1.0 for i in range(n)]
        traces.append(go.Scatter(x=samples, y=[v*100 for v in y], mode='lines', name=run.name))
    fig = go.Figure(data=traces)
    fig.update_layout(title='Latency CDF', xaxis_title='Total time (ms)', yaxis_title='Percentile')
    return fig

def build_accuracy_violin(runs: List[ModelRun]) -> go.Figure:
    # Order models by median F1 descending to highlight best performers first
    ordered = sorted_runs_by_metric(runs, 'f1', reverse=True)
    traces = []
    for run in ordered:
        scores = [entry.get('f1') for entry in run.per_question_summary if isinstance(entry.get('f1'), (int,float))]
        if not scores: continue
        traces.append(go.Violin(y=scores, name=run.name, box_visible=True, meanline_visible=True))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Per-question F1 Distribution',
        yaxis=dict(range=[0, 1], title='F1 (0–1)')
    )
    return fig

def build_answer_length_bubble(runs: List[ModelRun]) -> go.Figure:
    traces = []
    all_sizes = []
    per_model = []
    for run in runs:
        xs, ys, sizes, texts = [], [], [], []
        for entry in run.per_question_summary:
            length = entry.get('answer_length_chars'); score = entry.get('f1'); runtime = entry.get('total_time_ms')
            if not all(isinstance(v, (int,float)) for v in (length, score, runtime)): continue
            xs.append(length); ys.append(score); sizes.append(runtime); texts.append(f"Q{entry.get('id')}")
        if xs:
            per_model.append({'name': run.name, 'x': xs, 'y': ys, 'size': sizes, 'text': texts})
            all_sizes.extend(sizes)
    size_ref = (2.0 * max(all_sizes)) / (40.0 ** 2) if all_sizes else 1
    for item in per_model:
        traces.append(go.Scatter(x=item['x'], y=item['y'], mode='markers', name=item['name'],
                                 marker=dict(size=item['size'], sizemode='area', sizeref=size_ref, opacity=0.7),
                                 text=[f"{item['name']} · {t}" for t in item['text']],
                                 hovertemplate='%{text}<br>Chars: %{x}<br>F1: %{y:.3f}<br>Runtime: %{marker.size:.1f} ms<extra></extra>'))
    # Sort traces by their mean F1 descending for consistent ordering
    traces_sorted = sorted(traces, key=lambda t: (t.name or '').lower())
    fig = go.Figure(data=traces_sorted)
    fig.update_layout(title='Answer length vs Accuracy (bubble=size=runtime)', xaxis_title='Answer length (chars)', yaxis_title='F1 (0–1)')
    return fig

def build_prompt_vs_eval_scatter(runs: List[ModelRun]) -> go.Figure:
    # Order models by throughput for the scatter so higher throughput is on top
    ordered = sorted_runs_by_metric(runs, 'throughput_tokens_per_sec', reverse=True)
    traces = []
    for run in ordered:
        per_q = [e for e in run.per_question_summary if isinstance(e.get('prompt_eval_tokens'), (int,float)) and isinstance(e.get('eval_tokens'), (int,float))]
        if not per_q: continue
        traces.append(go.Scatter(x=[e['prompt_eval_tokens'] for e in per_q], y=[e['eval_tokens'] for e in per_q],
                                 mode='markers', name=run.name, text=[f"Q{e.get('id')}" for e in per_q], opacity=0.7,
                                 marker=dict(size=8), hovertemplate='Model: %{fullData.name}<br>%{text}<br>Prompt tokens: %{x}<br>Generated tokens: %{y}<extra></extra>'))
    fig = go.Figure(data=traces)
    fig.update_layout(title='Prompt vs Generated tokens', xaxis_title='Prompt tokens', yaxis_title='Generated tokens')
    return fig

def build_pareto_frontier(runs: List[ModelRun]) -> go.Figure:
    # Show throughput vs F1; sort by throughput for a readable pareto scatter.
    ordered = sorted_runs_by_metric(runs, 'throughput_tokens_per_sec', reverse=True)
    xs, ys, names = [], [], []
    for run in ordered:
        tp = get_metric(run.summary, 'throughput_tokens_per_sec')
        f1 = fetch_overall_metric(run.evaluation, 'f1')
        if tp is None or f1 is None: continue
        xs.append(tp); ys.append(f1); names.append(run.name)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode='markers+text', text=names, textposition='top center',
                               marker=dict(size=14, color=ys, colorscale='Viridis', showscale=True)))
    fig.update_layout(title='Pareto: Accuracy vs Throughput', xaxis_title='Throughput (tokens/sec)', yaxis_title='Overall F1')
    return fig

def build_time_per_token(runs: List[ModelRun]) -> go.Figure:
    ordered = sorted_runs_by_metric(runs, 'eval_time_ms', reverse=False)
    traces = []
    for run in ordered:
        samples = [entry.get('eval_time_ms') / entry.get('eval_tokens') for entry in run.per_question_summary
                   if isinstance(entry.get('eval_time_ms'), (int,float)) and isinstance(entry.get('eval_tokens'), (int,float)) and entry.get('eval_tokens')>0]
        if not samples: continue
        traces.append(go.Box(y=samples, name=run.name, boxmean='sd'))
    fig = go.Figure(data=traces)
    fig.update_layout(title='Time per Generated Token (ms/token)', yaxis_title='ms / token')
    return fig

def build_latency_histogram(runs: List[ModelRun]) -> go.Figure:
    ordered = sorted_runs_by_metric(runs, 'total_time_ms', reverse=False)
    traces = []
    for run in ordered:
        samples = [entry.get('total_time_ms') for entry in run.per_question_summary if isinstance(entry.get('total_time_ms'), (int,float))]
        if not samples: continue
        traces.append(go.Histogram(x=samples, name=run.name, opacity=0.55, nbinsx=30))
    fig = go.Figure(data=traces)
    fig.update_layout(title='Latency Histogram (per question)', xaxis_title='Latency (ms)', yaxis_title='Count', barmode='overlay')
    return fig

def build_graphs(runs: List[ModelRun]):
    # Generate concise inferences across runs (top models by core metrics)
    def safe_label(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"

    def compute_inferences(runs: List[ModelRun]) -> Dict[str, str]:
        # Collect per-model metrics
        rows = []
        for run in runs:
            rows.append({
                'name': run.name,
                'f1': fetch_overall_metric(run.evaluation, 'f1'),
                'exact_match': fetch_overall_metric(run.evaluation, 'exact_match'),
                'total_time_ms': get_metric(run.summary, 'total_time_ms'),
                'throughput': get_metric(run.summary, 'throughput_tokens_per_sec'),
                'memory_avg_mb': get_metric(run.summary, 'memory_avg_mb'),
            })
        # choose top/bottom
        def pick(metric, best='max'):
            filtered = [(r['name'], r[metric]) for r in rows if isinstance(r.get(metric), (int, float))]
            if not filtered:
                return ('n/a', None)
            if best == 'max':
                return max(filtered, key=lambda p: p[1])
            else:
                return min(filtered, key=lambda p: p[1])

        def pick_k(metric, k=3, best='max'):
            filtered = [(r['name'], r[metric]) for r in rows if isinstance(r.get(metric), (int, float))]
            if not filtered:
                return []
            filtered.sort(key=lambda p: p[1], reverse=(best == 'max'))
            return filtered[:k]
        best_f1_name, best_f1_val = pick('f1', 'max')
        best_em_name, best_em_val = pick('exact_match', 'max')
        fastest_name, fastest_val = pick('total_time_ms', 'min')
        top_throughput_name, top_throughput_val = pick('throughput', 'max')
        smallest_mem_name, smallest_mem_val = pick('memory_avg_mb', 'min')
        top3_by_f1 = pick_k('f1', 3, 'max')
        top3_by_throughput = pick_k('throughput', 3, 'max')
        top3_fastest = pick_k('total_time_ms', 3, 'min')

        # simple throughput per memory ratio ranking
        ratio = [(r['name'], r['throughput'] / r['memory_avg_mb']) for r in rows if isinstance(r.get('throughput'), (int, float)) and isinstance(r.get('memory_avg_mb'), (int, float))]
        best_ratio = max(ratio, key=lambda p: p[1]) if ratio else ('n/a', None)

        out = {}
        out['latency_accuracy'] = (
            f"Top F1: {', '.join([f'{n} ({v:.3f})' for n,v in top3_by_f1]) if top3_by_f1 else 'n/a'}; "
            f"Fastest models: {', '.join([f'{n} ({v:.0f} ms)' for n,v in top3_fastest]) if top3_fastest else 'n/a'}; "
            f"Top throughput: {', '.join([f'{n} ({v:.1f} tok/s)' for n,v in top3_by_throughput]) if top3_by_throughput else 'n/a'}."
        )
        out['latency_breakdown'] = (
            f"Fastest models: {', '.join([f'{n} ({v:.0f} ms)' for n,v in top3_fastest]) if top3_fastest else 'n/a'}. "
            f"Worst-loading model: {max(rows, key=lambda r: r.get('total_time_ms') or 0)['name'] if rows else 'n/a'}."
        )
        out['throughput'] = f"Top throughput: {', '.join([f'{n} ({v:.1f})' for n,v in top3_by_throughput]) if top3_by_throughput else 'n/a'} tok/s."
        out['accuracy'] = f"Top F1: {', '.join([f'{n} ({v:.3f})' for n,v in top3_by_f1]) if top3_by_f1 else 'n/a'}; Best EM: {best_em_name} ({safe_label(best_em_val)})."
        out['throughput_memory'] = (
            f"Best throughput/memory trade-off: {best_ratio[0]} ({best_ratio[1]:.3f} tok/s per MB) ; "
            f"Most memory efficient: {smallest_mem_name} ({safe_label(smallest_mem_val)} MB)."
        )
        # compute p90 per model and pick best
        def percentile(values, p):
            if not values: return None
            v_sorted = sorted(values)
            i = int(p/100.0 * (len(v_sorted)-1))
            return v_sorted[i]
        p90s = [(run.name, percentile([e.get('total_time_ms') for e in run.per_question_summary if isinstance(e.get('total_time_ms'), (int,float))], 90)) for run in runs]
        p90_filtered = [p for p in p90s if isinstance(p[1], (int,float))]
        p90_filtered.sort(key=lambda t: t[1])
        out['latency_cdf'] = f"Lowest p90 latency: {p90_filtered[0][0]} ({p90_filtered[0][1]:.1f} ms)" if p90_filtered else 'n/a'
        out['f1_distribution'] = f"Median F1 leader: {best_f1_name}; Top 3 by median F1: {', '.join([f'{n} ({v:.3f})' for n,v in top3_by_f1])}"
        out['pareto'] = f"Pareto suggestions: models near top-left are high-accuracy & high-throughput; top F1: {best_f1_name}"
        out['time_per_token'] = 'Lower ms/token is faster lexically; pick the model with the lowest median for fastest decoding.'
        out['latency_histogram'] = 'Compare skew and outliers; models with fatter right tails may have unpredictable tasks.'
        return out

    inferences = compute_inferences(runs)

    bundles = [
    ('Latency vs Accuracy', 'Avg total response time (ms) vs Overall F1 (0–1). Each point is a model. Sorted by F1 (high->low).\n' + inferences['latency_accuracy'], build_latency_vs_accuracy(runs)),
    ('Latency Breakdown', 'Stacked average stage durations per model in ms. Models sorted by average total time (fast->slow).\n' + inferences['latency_breakdown'], build_latency_breakdown(runs)),
    ('Throughput', 'Average end-to-end throughput (tokens/s). Models sorted by throughput (high->low).\n' + inferences['throughput'], build_throughput_line(runs)),
    ('Accuracy', 'Overall F1 and Exact Match per model. Bars grouped; models sorted by F1 (high->low).\n' + inferences['accuracy'], build_accuracy_bars(runs)),
    ('Throughput vs Memory', 'Throughput (tokens/s) vs average memory (MB). Shows speed vs footprint trade-offs.\n' + inferences['throughput_memory'], build_throughput_vs_memory(runs)),
    ('Latency CDF', 'Per-question latency percentiles (x=ms, y=percentile). Curves ordered by avg latency (fast->slow).\n' + inferences['latency_cdf'], build_latency_cdf(runs)),
    ('F1 Distribution', 'Per-question F1 violin plots (0–1); models ordered by median F1 (best->worst).\n' + inferences['f1_distribution'], build_accuracy_violin(runs)),
    ('Pareto Frontier', 'Throughput (tokens/s) vs Overall F1 scatter. Useful to find speed/quality trade-offs.\n' + inferences['pareto'], build_pareto_frontier(runs)),
    ('Time per Token', 'Per question decoder speed (ms/token) distribution per model; lower = faster.\n' + inferences['time_per_token'], build_time_per_token(runs)),
        ('Latency Histogram', 'Per-question latency distribution; helpful to inspect skew and outliers.\n' + inferences['latency_histogram'], build_latency_histogram(runs)),
    ]
    return bundles

def export_dashboard(bundles, destination: Path):
    snippets = []
    for title, desc, fig in bundles:
        figure_html = fig.to_html(full_html=False, include_plotlyjs=False)
        snippets.append('<section class="card"><h2>' + title + '</h2><p class="description">' + desc + '</p>' + figure_html + '</section>')
    head = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<title>LLM Benchmark Dashboard</title>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<style>
body { font-family: Inter, Arial, sans-serif; margin: 0; padding: 2rem; background: #f6f8fb; }
h1 { text-align: center; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }
.card { background: #fff; border-radius: 12px; padding: 14px; box-shadow: 0 6px 20px rgba(10,20,30,0.06); }
.description { color: #374151; margin-top: 0; }
</style>
</head>
<body>
<h1>LLM Benchmark Dashboard</h1>
<div class='grid'>"""
    tail = """</div>
</body>
</html>"""
    html_output = head + ''.join(snippets) + tail
    destination.write_text(html_output, encoding='utf-8')
    print('Wrote dashboard to', destination)

def main(root_path: Path, out_path: Path):
    runs = collect_model_runs(root_path)
    if not runs:
        raise SystemExit('No model runs found in {}'.format(root_path))
    bundles = build_graphs(runs)
    export_dashboard(bundles, out_path)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='root dir with model folders')
    p.add_argument('--out', default='dashboard.html', help='output html file')
    args = p.parse_args()
    main(Path(args.root).resolve(), Path(args.out).resolve())
