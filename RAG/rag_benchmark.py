#!/usr/bin/env python3
"""End-to-end RAG benchmark for resume-driven QA workloads.

This script is self-contained and does not rely on other modules inside the
repository. It performs the following high-level steps:

1. Load ``resume.pdf`` (or any TXT/DOCX) and build a TF-IDF retriever over
   overlapping text chunks.
2. Iterate over ``dataset.json`` (10 question/answer pairs) and, for each
   question, measure retrieval latency/carbon, collect the top-k chunks, and
   augment the prompt for llama.cpp.
3. Run llama.cpp with the augmented prompt while capturing stdout/stderr,
   memory usage, latency, throughput, and CodeCarbon emissions.
4. Compute EM/F1 plus TF-IDF cosine semantic similarity against the gold
   answers, write raw/parsed outputs, evaluation metrics, summary metrics, and
   an additional ``prompt_log.json`` for auditability.
5. Emit a ``summary.json`` that condenses the benchmark run.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:  # Optional dependency for carbon accounting
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:  # pragma: no cover - CodeCarbon may not be installed
    EmissionsTracker = None  # type: ignore

try:
    import PyPDF2
except Exception:  # pragma: no cover - loaded lazily when needed
    PyPDF2 = None  # type: ignore

try:
    import docx2txt
except Exception:  # pragma: no cover
    docx2txt = None  # type: ignore

SYSTEM_PROMPT = (
    "You are answering questions about Sai Kapil Bharadwaj's resume. "
    "Use only the provided resume context, cite verbatim facts, start the final "
    "answer with ***ANSWER***, and keep responses concise."
)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    body: str

    def render(self, context: str, question: str) -> str:
        return self.body.format(system=SYSTEM_PROMPT, context=context.strip(), question=question.strip())


PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "llama": PromptTemplate(
        name="LLaMA",
        body=(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "{system}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "You will answer a question using only the provided resume context. "
            "Return the short answer prefixed with ***ANSWER***.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    ),
    "qwen": PromptTemplate(
        name="Qwen",
        body=(
            "<|im_start|>system\n"
            "{system}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n"
            "Respond with ***ANSWER*** followed by the span from the context.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    ),
    "gemma": PromptTemplate(
        name="Gemma",
        body=(
            "<start_of_turn>user\n"
            "{system}\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
    ),
}


def detect_model_family(model_path: str) -> str:
    lowered = model_path.lower()
    if "gemma" in lowered:
        return "gemma"
    if "qwen" in lowered:
        return "qwen"
    return "llama"


def derive_model_slug(model_path: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", Path(model_path).name)


def build_llama_command(cli_path: str, model_path: str, prompt: str) -> List[str]:
    return [
        cli_path,
        "-m",
        model_path,
        "-p",
        prompt,
        "--single-turn",
        "-no-cnv",
        "--seed",
        "0",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--n-predict",
        "256",
        "--ctx-size",
        "2048",
        "--threads",
        "6",
        "--threads-batch",
        "6",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--no-warmup",
    ]


def load_file(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".pdf"):
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 is required to read PDF files")
        text = []
        with open(path, "rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            for page in reader.pages:
                chunk = page.extract_text() or ""
                text.append(chunk)
        return "\n".join(text)
    if lowered.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    if lowered.endswith(".docx"):
        if docx2txt is None:
            raise RuntimeError("docx2txt is required to read DOCX files")
        return docx2txt.process(path)
    raise ValueError("Unsupported file type. Use PDF/TXT/DOCX.")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return chunks


def start_carbon_tracker(project_name: str) -> Optional[Any]:
    if EmissionsTracker is None:
        return None
    try:
        tracker = EmissionsTracker(project_name=project_name, save_to_file=False)
        tracker.start()
        return tracker
    except Exception:  # pragma: no cover
        return None


def stop_carbon_tracker(tracker: Optional[Any]) -> Optional[float]:
    if tracker is None:
        return None
    try:
        return tracker.stop()
    except Exception:  # pragma: no cover
        return None


class MemoryMonitor:
    """Poll /proc/<pid>/status for RSS samples while the subprocess runs."""

    def __init__(self, pid: int, poll_interval: float = 0.1) -> None:
        self.pid = pid
        self.poll_interval = poll_interval
        self.samples: List[Dict[str, float]] = []
        self.peak_kb = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _poll(self) -> None:
        status_path = Path("/proc") / str(self.pid) / "status"
        if not status_path.exists():
            return
        while not self._stop_event.is_set():
            rss_kb = self._read_rss(status_path)
            if rss_kb is None:
                break
            timestamp = time.time()
            self.samples.append({"timestamp": timestamp, "rss_kb": rss_kb})
            self.peak_kb = max(self.peak_kb, rss_kb)
            time.sleep(self.poll_interval)

    @staticmethod
    def _read_rss(status_path: Path) -> Optional[int]:
        try:
            with status_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1])
        except FileNotFoundError:
            return None
        return None


def run_single_example(command: Sequence[str]) -> Tuple[str, str, Dict[str, Any]]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    monitor = MemoryMonitor(process.pid)
    monitor.start()

    try:
        stdout_data, stderr_data = process.communicate()
    finally:
        monitor.stop()

    if process.returncode != 0:
        stderr_tail = "\n".join(stderr_data.strip().splitlines()[-10:]) if stderr_data else "<no stderr captured>"
        raise RuntimeError(
            "Command failed with exit code "
            f"{process.returncode}. Last stderr lines:\n{stderr_tail}"
        )

    return (
        stdout_data,
        stderr_data,
        {
            "memory_usage_mb": monitor.peak_kb / 1024 if monitor.peak_kb else None,
            "memory_samples": monitor.samples,
            "carbon_emissions_kg": None,
        },
    )


SAMPLING_RE = re.compile(r"sampling time =\s*([0-9.]+) ms /\s*([0-9]+) runs")
LOAD_RE = re.compile(r"load time =\s*([0-9.]+) ms")
PROMPT_RE = re.compile(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
EVAL_RE = re.compile(r"eval time =\s*([0-9.]+) ms /\s*([0-9]+) (?:runs|tokens)")
TOTAL_RE = re.compile(r"total time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
GRAPHS_RE = re.compile(r"graphs reused =\s*([0-9]+)")


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def extract_perf_lines(stderr_text: str) -> List[str]:
    prefixes = ("llama_perf_sampler_print:", "llama_perf_context_print:")
    return [line.strip() for line in stderr_text.splitlines() if line.strip().startswith(prefixes)]


def parse_metrics(perf_lines: Iterable[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for line in perf_lines:
        if "sampling time" in line:
            match = SAMPLING_RE.search(line)
            if match:
                total_ms = float(match.group(1))
                runs = int(match.group(2))
                metrics["sampling_time"] = {
                    "total_sampling_time_ms": total_ms,
                    "total_sampling_runs": runs,
                    "time_per_token_ms": safe_ratio(total_ms, runs),
                    "tokens_per_sec": safe_ratio(runs * 1000.0, total_ms),
                }
        elif "prompt eval time" in line:
            match = PROMPT_RE.search(line)
            if match:
                total_ms = float(match.group(1))
                tokens = int(match.group(2))
                metrics["prompt_eval_time"] = {
                    "total_prompt_eval_time_ms": total_ms,
                    "total_prompt_eval_tokens": tokens,
                    "time_per_token_ms": safe_ratio(total_ms, tokens),
                    "tokens_per_sec": safe_ratio(tokens * 1000.0, total_ms),
                }
        elif "eval time" in line:
            match = EVAL_RE.search(line)
            if match:
                total_ms = float(match.group(1))
                tokens = int(match.group(2))
                metrics["eval_time"] = {
                    "total_eval_time_ms": total_ms,
                    "total_eval_tokens": tokens,
                    "time_per_token_ms": safe_ratio(total_ms, tokens),
                    "tokens_per_sec": safe_ratio(tokens * 1000.0, total_ms),
                }
        elif "load time" in line:
            match = LOAD_RE.search(line)
            if match:
                metrics["load_time_ms"] = float(match.group(1))
        elif "total time" in line:
            match = TOTAL_RE.search(line)
            if match:
                metrics["total_time_ms"] = float(match.group(1))
                metrics["total_tokens"] = int(match.group(2))
        elif "graphs reused" in line:
            match = GRAPHS_RE.search(line)
            if match:
                metrics["graphs_reused"] = int(match.group(1))
    return metrics


def trim_stdout_to_end_token(stdout_text: str, end_token: str = "[end of text]") -> str:
    end_idx = stdout_text.find(end_token)
    usable = stdout_text if end_idx == -1 else stdout_text[:end_idx]
    if "***ANSWER***" not in usable:
        return usable.strip()
    segments = usable.split("***ANSWER***")
    for segment in reversed(segments[1:]):
        stripped = segment.strip()
        if stripped:
            return stripped
    return usable.strip()


def summarize_memory_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    values_kb: List[float] = []
    for sample in samples:
        rss = sample.get("rss_kb") if isinstance(sample, dict) else None
        if isinstance(rss, (int, float)):
            values_kb.append(float(rss))
    if not values_kb:
        return {"memory_min_mb": None, "memory_max_mb": None, "memory_avg_mb": None}
    min_mb = min(values_kb) / 1024.0
    max_mb = max(values_kb) / 1024.0
    avg_mb = (sum(values_kb) / len(values_kb)) / 1024.0
    return {"memory_min_mb": min_mb, "memory_max_mb": max_mb, "memory_avg_mb": avg_mb}


WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^0-9a-zA-Z]+")


def normalize_answer(answer: str) -> str:
    lowered = answer.lower().strip()
    no_punct = PUNCT_RE.sub(" ", lowered)
    collapsed = WHITESPACE_RE.sub(" ", no_punct)
    return collapsed.strip()


def tokenize(answer: str) -> List[str]:
    normalized = normalize_answer(answer)
    return normalized.split() if normalized else []


def exact_match(prediction: str, gold_answers: Sequence[str]) -> int:
    norm_pred = normalize_answer(prediction)
    for gold in gold_answers:
        if norm_pred == normalize_answer(gold):
            return 1
    return 0


def overlap_counts(pred_tokens: List[str], gold_tokens: List[str]) -> int:
    freq: Dict[str, int] = {}
    for token in gold_tokens:
        freq[token] = freq.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if freq.get(token, 0) > 0:
            overlap += 1
            freq[token] -= 1
    return overlap


def safe_f1(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = overlap_counts(pred_tokens, gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    pred_tokens = tokenize(prediction)
    gold_lists = [tokenize(ans) for ans in gold_answers]
    if not gold_lists:
        return float(not pred_tokens)
    scores = [safe_f1(pred_tokens, tokens) for tokens in gold_lists]
    return max(scores)


def evaluate_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    em_scores: List[int] = []
    f1_scores: List[float] = []
    carbon_values: List[float] = []
    per_question: List[Dict[str, Any]] = []

    for entry in entries:
        prediction = entry.get("model_answer", "")
        gold_answers = entry.get("gold_answers", [])
        if not isinstance(gold_answers, list):
            gold_answers = []
        em = exact_match(prediction, gold_answers)
        f1 = best_f1(prediction, gold_answers)
        carbon = entry.get("carbon_emissions_kg")

        entry["exact_match"] = em
        entry["f1"] = f1

        em_scores.append(em)
        f1_scores.append(f1)
        if isinstance(carbon, (int, float)):
            carbon_values.append(float(carbon))

        per_question.append(
            {
                "id": entry.get("id"),
                "question": entry.get("question"),
                "model_answer": prediction,
                "gold_answers": gold_answers,
                "exact_match": em,
                "f1": f1,
                "carbon_emissions_kg": carbon,
            }
        )

    overall_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    total_carbon = sum(carbon_values) if carbon_values else None
    avg_carbon = (total_carbon / len(carbon_values)) if carbon_values else None

    overall: Dict[str, Any] = {
        "exact_match": overall_em,
        "f1": overall_f1,
        "question_count": len(entries),
    }
    if total_carbon is not None and avg_carbon is not None:
        overall["carbon_emissions_kg_sum"] = total_carbon
        overall["carbon_emissions_kg_avg"] = avg_carbon

    return {"overall": overall, "per_question": per_question}


Number = Optional[float]


def nested_get(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return float(current) if isinstance(current, (int, float)) else None


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


SERIES_SPECS: Dict[str, Any] = {
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
    prompt_section = entry.get("prompt_eval_time") or {}
    eval_section = entry.get("eval_time") or {}
    sampling_section = entry.get("sampling_time") or {}
    throughput = compute_throughput(entry)
    return {
        "id": entry.get("id"),
        "total_time_ms": entry.get("total_time_ms"),
        "load_time_ms": entry.get("load_time_ms"),
        "prompt_eval_time_ms": prompt_section.get("total_prompt_eval_time_ms"),
        "prompt_eval_tokens": prompt_section.get("total_prompt_eval_tokens"),
        "eval_time_ms": eval_section.get("total_eval_time_ms"),
        "eval_tokens": eval_section.get("total_eval_tokens"),
        "sampling_time_ms": sampling_section.get("total_sampling_time_ms"),
        "sampling_runs": sampling_section.get("total_sampling_runs"),
        "total_tokens": entry.get("total_tokens"),
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
    return {
        "question_count": len(entries),
        "metrics": metrics,
        "per_question": [build_per_question(entry) for entry in entries],
    }


@dataclass
class RetrievalResult:
    latency_ms: float
    carbon_kg: Optional[float] = None
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class RetrievalEngine:
    def __init__(self, chunks: Sequence[str], vectorizer: TfidfVectorizer, matrix, top_k: int) -> None:
        self.chunks = list(chunks)
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.top_k = top_k

    @classmethod
    def from_document(cls, text: str, chunk_size: int, overlap: int, top_k: int) -> "RetrievalEngine":
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(chunks)
        return cls(chunks, vectorizer, matrix, top_k)

    def query(self, question: str, label: str) -> RetrievalResult:
        start = time.perf_counter()
        query_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = sims.argsort()[::-1][: self.top_k]
        latency_ms = (time.perf_counter() - start) * 1000.0
        chunks: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunks.append(
                {
                    "rank": rank,
                    "chunk_index": int(idx),
                    "similarity": float(sims[idx]),
                    "text": self.chunks[idx].strip(),
                }
            )
        return RetrievalResult(latency_ms=latency_ms, chunks=chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG benchmark over resume questions")
    parser.add_argument("--resume", default="resume.pdf", help="Path to the resume document")
    parser.add_argument("--questions", default="dataset.json", help="Path to the question/answer JSON list")
    parser.add_argument("--model", required=True, help="Path to the GGUF model (relative or absolute)")
    parser.add_argument("--llama-cli", required=True, help="Path to the llama.cpp CLI executable")
    parser.add_argument("--output-dir", default="runs", help="Directory where benchmark artifacts are stored")
    parser.add_argument("--chunk-size", type=int, default=500, help="Character length per resume chunk")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap between successive chunks")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks per question")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional limit for debugging")
    return parser.parse_args()


def load_questions(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected the questions file to contain a list of entries")
    return data[:limit] if limit is not None else data


def render_context(chunks: Sequence[Dict[str, Any]]) -> str:
    if not chunks:
        return ""
    sections: List[str] = []
    for chunk in chunks:
        score = chunk.get("similarity")
        prefix = f"[Rank {chunk.get('rank')} | score={score:.4f}]" if isinstance(score, (int, float)) else ""
        sections.append(f"{prefix}\n{chunk.get('text', '')}".strip())
    return "\n\n".join(sections).strip()


def semantic_similarity(prediction: str, reference: str) -> float:
    texts = [prediction or "", reference or ""]
    if not any(texts):
        return 1.0
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return 0.0
    if matrix.shape[0] < 2:
        return 0.0
    similarity = cosine_similarity(matrix[0], matrix[1])[0][0]
    return float(similarity)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("rag_benchmark")

    model_slug = derive_model_slug(args.model)
    model_family = detect_model_family(args.model)
    template = PROMPT_TEMPLATES[model_family]
    session_tracker = start_carbon_tracker(f"rag_session_{model_slug}")

    resume_path = Path(args.resume).resolve()
    questions_path = Path(args.questions).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading resume from %s", resume_path)
    resume_text = load_file(str(resume_path))

    logger.info(
        "Building TF-IDF retriever | chunk_size=%s overlap=%s top_k=%s",
        args.chunk_size,
        args.chunk_overlap,
        args.top_k,
    )
    retriever = RetrievalEngine.from_document(
        resume_text,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        top_k=args.top_k,
    )

    logger.info("Loading questions from %s", questions_path)
    questions = load_questions(questions_path, args.max_questions)
    logger.info("Loaded %d questions", len(questions))

    model_output_dir = output_root / model_slug
    model_output_dir.mkdir(parents=True, exist_ok=True)

    raw_outputs: List[Dict[str, Any]] = []
    parsed_outputs: List[Dict[str, Any]] = []
    raw_metrics_payload: Dict[str, Any] = {"model": args.model, "model_slug": model_slug, "entries": []}
    prompt_log: List[Dict[str, Any]] = []

    retrieval_latencies: List[float] = []
    semantic_scores: List[float] = []

    for entry in questions:
        question_id = int(entry.get("id", len(parsed_outputs)))
        question = entry.get("question", "").strip()
        gold_answer = entry.get("answer") or entry.get("answers")
        if isinstance(gold_answer, list):
            gold_answers = [str(ans) for ans in gold_answer]
        else:
            gold_answers = [str(gold_answer)] if gold_answer else []
        logger.info("[Q%02d] Retrieving context", question_id)
        retrieval_tracker = start_carbon_tracker(f"retrieval_q{question_id}_{model_slug}")
        retrieval = retriever.query(question, label=f"q{question_id}")
        retrieval_carbon = stop_carbon_tracker(retrieval_tracker)
        retrieval.carbon_kg = retrieval_carbon
        context = render_context(retrieval.chunks)
        prompt = template.render(context=context, question=question)
        prompt_log.append(
            {
                "id": question_id,
                "question": question,
                "retrieval_latency_ms": retrieval.latency_ms,
                "retrieved_chunks": retrieval.chunks,
                "augmented_prompt": prompt,
            }
        )

        command = build_llama_command(args.llama_cli, args.model, prompt)

        logger.info("[Q%02d] Starting llama.cpp inference", question_id)
        generation_tracker = start_carbon_tracker(f"generation_q{question_id}_{model_slug}")
        generation_carbon: Optional[float] = None
        try:
            stdout_data, stderr_data, memory_blob = run_single_example(command)
        except Exception as exc:  # noqa: BLE001
            generation_carbon = stop_carbon_tracker(generation_tracker)
            logger.error("[Q%02d] FAILED: %s", question_id, exc)
            failure_payload = {
                "id": question_id,
                "question": question,
                "error": str(exc),
                "prompt": prompt,
                "generation_carbon_kg": generation_carbon,
            }
            raw_outputs.append(failure_payload)
            parsed_outputs.append(failure_payload)
            continue

        generation_carbon = stop_carbon_tracker(generation_tracker)

        trimmed_answer = trim_stdout_to_end_token(stdout_data)
        perf_lines = extract_perf_lines(stderr_data)
        memory_stats = summarize_memory_samples(memory_blob.get("memory_samples", []))
        metrics = parse_metrics(perf_lines)

        em = exact_match(trimmed_answer, gold_answers)
        f1 = best_f1(trimmed_answer, gold_answers)
        semantic_score = semantic_similarity(trimmed_answer, " ".join(gold_answers)) if gold_answers else 0.0

        retrieval_latencies.append(retrieval.latency_ms)
        semantic_scores.append(semantic_score)

        memory_blob["carbon_emissions_kg"] = generation_carbon

        raw_item = {
            "id": question_id,
            "question": question,
            "gold_answers": gold_answers,
            "prompt": prompt,
            "model_answer": stdout_data,
            "stderr": "\n".join(perf_lines),
            "retrieved_chunks": retrieval.chunks,
            "retrieval_latency_ms": retrieval.latency_ms,
            "retrieval_carbon_kg": retrieval.carbon_kg,
            "generation_carbon_kg": generation_carbon,
            **memory_blob,
        }
        raw_outputs.append(raw_item)

        parsed_item = {
            **{k: v for k, v in raw_item.items() if k != "memory_samples"},
            "model_answer": trimmed_answer,
            **memory_stats,
            **metrics,
            "retrieval_latency_ms": retrieval.latency_ms,
            "retrieval_carbon_kg": retrieval.carbon_kg,
            "generation_carbon_kg": generation_carbon,
            "exact_match": em,
            "f1": f1,
            "semantic_similarity": semantic_score,
        }
        parsed_outputs.append(parsed_item)

        raw_metrics_payload["entries"].append(
            {
                "id": question_id,
                "stderr_lines": perf_lines,
                "retrieval_latency_ms": retrieval.latency_ms,
            }
        )

        logger.info(
            "[Q%02d] Done | retrieval=%.2f ms | answer_chars=%d | EM=%d | F1=%.3f",
            question_id,
            retrieval.latency_ms,
            len(trimmed_answer),
            em,
            f1,
        )

    raw_path = model_output_dir / f"raw_outputs_{model_slug}.json"
    parsed_path = model_output_dir / f"parsed_outputs_{model_slug}.json"
    metrics_path = model_output_dir / "raw_metrics.json"
    eval_path = model_output_dir / "evaluation_metrics.json"
    summary_metrics_path = model_output_dir / "summary_metrics.json"
    prompt_log_path = model_output_dir / "prompt_log.json"
    summary_json_path = model_output_dir / "summary.json"

    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(raw_outputs, handle, indent=2, ensure_ascii=False)

    evaluation_payload = evaluate_entries(parsed_outputs)
    evaluation_payload.update(
        {
            "model_directory": model_output_dir.name,
            "source_file": parsed_path.name,
            "root": str(model_output_dir.resolve()),
        }
    )

    with parsed_path.open("w", encoding="utf-8") as handle:
        json.dump(parsed_outputs, handle, indent=2, ensure_ascii=False)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(raw_metrics_payload, handle, indent=2, ensure_ascii=False)

    with eval_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_payload, handle, indent=2, ensure_ascii=False)

    with prompt_log_path.open("w", encoding="utf-8") as handle:
        json.dump(prompt_log, handle, indent=2, ensure_ascii=False)

    summary_metrics_payload = summarize_entries(parsed_outputs)
    summary_metrics_payload.update(
        {
            "model_directory": model_output_dir.name,
            "source_file": parsed_path.name,
            "root": str(model_output_dir.resolve()),
        }
    )
    with summary_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_metrics_payload, handle, indent=2, ensure_ascii=False)

    session_emissions = stop_carbon_tracker(session_tracker)

    summary_payload = {
        "model": args.model,
        "model_slug": model_slug,
        "question_count": len(parsed_outputs),
        "retrieval_latency_ms": {
            "avg": sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else None,
            "min": min(retrieval_latencies) if retrieval_latencies else None,
            "max": max(retrieval_latencies) if retrieval_latencies else None,
        },
        "semantic_similarity_avg": sum(semantic_scores) / len(semantic_scores) if semantic_scores else None,
        "session_carbon_emissions_kg": session_emissions,
        "evaluation": evaluation_payload.get("overall"),
        "artifacts": {
            "raw_outputs": str(raw_path),
            "parsed_outputs": str(parsed_path),
            "raw_metrics": str(metrics_path),
            "evaluation_metrics": str(eval_path),
            "summary_metrics": str(summary_metrics_path),
            "prompt_log": str(prompt_log_path),
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    logger.info("Saved raw outputs to %s", raw_path)
    logger.info("Saved parsed outputs to %s", parsed_path)
    logger.info("Saved raw metrics to %s", metrics_path)
    logger.info("Saved evaluation metrics to %s", eval_path)
    logger.info("Saved summary metrics to %s", summary_metrics_path)
    logger.info("Saved prompt log to %s", prompt_log_path)
    logger.info("Saved condensed summary to %s", summary_json_path)


if __name__ == "__main__":
    main()
