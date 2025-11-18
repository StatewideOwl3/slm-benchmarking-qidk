#!/usr/bin/env python3
"""Run the llama.cpp CLI over the SQuAD-50 benchmark and capture metrics.

This script is designed to be launched from the root of a llama.cpp checkout.
It executes the command described in ``qna_instructions.md`` for each of the 50
SQuAD entries, records raw/stdout outputs, captures profiling metrics from
stderr, and measures peak memory consumption using OS calls.

Outputs
=======

* ``raw_outputs_<model>.json`` – verbatim stdout/stderr plus prompt metadata.
* ``parsed_outputs_<model>.json`` – structured metrics derived from stderr.
* ``raw_metrics.json`` – raw profiling lines kept separately for auditing.

The script relies solely on the system utilities provided by Termux/Linux and
therefore avoids third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional dependency for carbon tracking
	from codecarbon import EmissionsTracker  # type: ignore
except Exception:  # pragma: no cover - CodeCarbon may not be installed
	EmissionsTracker = None  # type: ignore


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
	"You are a grounded question answering model that is currently undergoing "
	"an extractive Q&A benchmark (SQuAD-50). Keep every answer precise, cite "
	"only what is supported by the provided context, and never guess. To help "
	"with downstream evaluation, always demarcate the start of your final "
	"response with the literal token ***ANSWER*** followed immediately by the "
	"short answer. Do not include commentary before that marker."
)


@dataclass(frozen=True)
class PromptTemplate:
	"""Container for chat-style prompt templates."""

	name: str
	body: str

	def render(self, context: str, question: str) -> str:
		"""Backfill the template placeholders with the current item."""

		return self.body.format(system=SYSTEM_PROMPT, context=context.strip(), question=question.strip())


PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
	# Meta Llama / llama.cpp default chat template using the INST blocks.
	"llama": PromptTemplate(
		name="LLaMA",
		body=(
			"<|start_header_id|>system<|end_header_id|>\n\n"
			"{system}\n"
			"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
			"The following passage and question are from the SQuAD benchmark.\n"
			"Return only the answer span from the context and begin with ***ANSWER***.\n\n"
			"Context:\n{context}\n\nQuestion:\n{question}\n"
			"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
		),
	),
	# Qwen uses <|im_start|> and <|im_end|> markers with explicit roles.
	"qwen": PromptTemplate(
		name="Qwen",
		body=(
			"<|im_start|>system\n"
			"{system}\n"
			"<|im_end|>\n"
			"<|im_start|>user\n"
			"You will be given a context paragraph and a question.\n"
			"Provide the precise span, prefixed with ***ANSWER***.\n\n"
			"Context:\n{context}\n\nQuestion:\n{question}\n"
			"<|im_end|>\n"
			"<|im_start|>assistant\n"
		),
	),
	# Gemma chat template follows the start/end-of-turn markers from the docs.
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


# ---------------------------------------------------------------------------
# Memory monitoring helpers
# ---------------------------------------------------------------------------


class MemoryMonitor:
	"""Poll /proc for RSS usage while a subprocess is running."""

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
			return  # /proc is unavailable on this platform.
		while not self._stop_event.is_set():
			rss_kb = self._read_rss(status_path)
			if rss_kb is None:
				break
			timestamp = time.time()
			self.samples.append({"timestamp": timestamp, "rss_kb": rss_kb})
			if rss_kb > self.peak_kb:
				self.peak_kb = rss_kb
			time.sleep(self.poll_interval)

	@staticmethod
	def _read_rss(status_path: Path) -> Optional[int]:
		try:
			with status_path.open("r", encoding="utf-8") as handle:
				for line in handle:
					if line.startswith("VmRSS:"):
						parts = line.split()
						if len(parts) >= 2:
							return int(parts[1])  # Value is already in KiB.
		except FileNotFoundError:
			return None
		return None


# ---------------------------------------------------------------------------
# Metric parsing helpers
# ---------------------------------------------------------------------------


SAMPLING_RE = re.compile(r"sampling time =\s*([0-9.]+) ms /\s*([0-9]+) runs")
LOAD_RE = re.compile(r"load time =\s*([0-9.]+) ms")
PROMPT_RE = re.compile(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
EVAL_RE = re.compile(r"eval time =\s*([0-9.]+) ms /\s*([0-9]+) (?:runs|tokens)")
TOTAL_RE = re.compile(r"total time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
GRAPHS_RE = re.compile(r"graphs reused =\s*([0-9]+)")


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
	"""Return ``numerator/denominator`` guarding against division by zero."""

	if denominator == 0:
		return None
	return numerator / denominator


def extract_perf_lines(stderr_text: str) -> List[str]:
	"""Capture the llama_perf / llama_memory lines verbatim for auditing."""

	prefixes = (
		"llama_perf_sampler_print:",
		"llama_perf_context_print:",
	)
	return [line.strip() for line in stderr_text.splitlines() if line.strip().startswith(prefixes)]


def parse_metrics(perf_lines: Iterable[str]) -> Dict[str, object]:
	"""Parse the profiling metrics emitted by llama.cpp."""

	metrics: Dict[str, object] = {}
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


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the llama.cpp Q&A benchmark")
	parser.add_argument(
		"--model",
		required=True,
		help="Path to the GGUF model relative to the llama.cpp directory",
	)
	parser.add_argument(
		"--questions",
		default="squad_50.json",
		help="Path to the SQuAD-style JSON file (default: squad_50.json)",
	)
	parser.add_argument(
		"--llama-cli",
		default="./build/bin/llama-cli",
		help="Path to the llama.cpp CLI binary",
	)
	parser.add_argument(
		"--output-dir",
		default=".",
		help="Directory where JSON artifacts will be written",
	)
	parser.add_argument(
		"--max-questions",
		type=int,
		default=None,
		help="Optional cap for debugging; process only the first N items",
	)
	return parser.parse_args()


def load_questions(path: Path) -> List[Dict[str, object]]:
	with path.open("r", encoding="utf-8") as handle:
		data = json.load(handle)
	if not isinstance(data, list):
		raise ValueError("Expected the questions file to contain a list of entries")
	return data


def detect_model_family(model_path: str) -> str:
	lowered = model_path.lower()
	if "gemma" in lowered:
		return "gemma"
	if "qwen" in lowered:
		return "qwen"
	return "llama"


def derive_model_slug(model_path: str) -> str:
	name = Path(model_path).name
	return re.sub(r"[^0-9A-Za-z_.-]", "_", name)


def build_command(cli_path: str, model_path: str, prompt: str) -> List[str]:
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


def trim_stdout_to_end_token(stdout_text: str, end_token: str = "[end of text]") -> str:
	"""Return only the content between the LAST ***ANSWER*** marker and the end token."""

	end_idx = stdout_text.find(end_token)
	usable_text = stdout_text if end_idx == -1 else stdout_text[:end_idx]
	if "***ANSWER***" not in usable_text:
		return usable_text.strip()

	segments = usable_text.split("***ANSWER***")
	for segment in reversed(segments[1:]):
		stripped = segment.strip()
		if stripped:
			return stripped
	# If every segment after a marker is blank, fall back to the portion before the first marker.
	return segments[0].strip() or usable_text.strip()


def summarize_memory_samples(samples: Sequence[Dict[str, object]]) -> Dict[str, Optional[float]]:
	"""Compute min/avg/max memory usage (in MB) from the RSS sample list."""

	values_kb: List[float] = []
	for sample in samples:
		rss_value = sample.get("rss_kb") if isinstance(sample, dict) else None
		if isinstance(rss_value, (int, float)):
			values_kb.append(float(rss_value))
	if not values_kb:
		return {
			"memory_min_mb": None,
			"memory_max_mb": None,
			"memory_avg_mb": None,
		}

	min_mb = min(values_kb) / 1024.0
	max_mb = max(values_kb) / 1024.0
	avg_mb = (sum(values_kb) / len(values_kb)) / 1024.0
	return {
		"memory_min_mb": min_mb,
		"memory_max_mb": max_mb,
		"memory_avg_mb": avg_mb,
	}


def pretty_print_block(question_id: int, title: str, content: str) -> None:
	"""Print a small labeled block to the terminal to reduce clutter."""

	label = f"[Q{question_id:02d}] --- {title} ---"
	text = content if isinstance(content, str) else str(content)
	print(label, flush=True)
	print(text.strip() if text.strip() else "<empty>", flush=True)
	print(f"[Q{question_id:02d}] --- END {title} ---", flush=True)


def run_single_example(
	command: Sequence[str],
	capture_memory: bool = True,
) -> Tuple[str, str, Dict[str, object]]:
	"""Run the llama.cpp CLI once and capture stdout, stderr, and memory data."""

	tracker: Optional[object] = None
	emissions_kg: Optional[float] = None
	if EmissionsTracker is not None:
		try:
			tracker = EmissionsTracker(
				measure_power_secs=1,
				save_to_file=False,
				tracking_mode="machine",
				log_level="error",
			)
		except Exception:
			tracker = None

	process = subprocess.Popen(
		command,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		encoding="utf-8",
		errors="replace",
	)

	if tracker is not None:
		try:
			tracker.start()
		except Exception:
			tracker = None

	monitor: Optional[MemoryMonitor] = None
	if capture_memory:
		monitor = MemoryMonitor(process.pid)
		monitor.start()

	try:
		stdout_data, stderr_data = process.communicate()
	finally:
		if monitor:
			monitor.stop()
		if tracker is not None:
			try:
				emissions_kg = tracker.stop()
			except Exception:
				emissions_kg = None
			tracker = None

	mem_stats = {"memory_usage_mb": None, "memory_samples": []}
	if monitor:
		mem_stats["memory_usage_mb"] = monitor.peak_kb / 1024 if monitor.peak_kb else None
		mem_stats["memory_samples"] = monitor.samples
	mem_stats["carbon_emissions_kg"] = emissions_kg

	if process.returncode != 0:
		raise RuntimeError(f"Command failed with exit code {process.returncode}:")

	return stdout_data, stderr_data, mem_stats


def main() -> None:
	args = parse_args()
	questions_path = Path(args.questions)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	questions = load_questions(questions_path)
	if args.max_questions:
		questions = questions[: args.max_questions]

	model_family = detect_model_family(args.model)
	template = PROMPT_TEMPLATES[model_family]
	model_slug = derive_model_slug(args.model)

	raw_outputs: List[Dict[str, object]] = []
	parsed_outputs: List[Dict[str, object]] = []
	raw_metrics_payload: Dict[str, object] = {
		"model": args.model,
		"model_slug": model_slug,
		"entries": [],
	}

	for entry in questions:
		question_id = int(entry.get("id"))
		context = entry.get("context", "")
		question = entry.get("question", "")
		gold_answers = entry.get("answers", [])

		prompt = template.render(context=context, question=question)
		command = build_command(args.llama_cli, args.model, prompt)

		print(f"[Q{question_id:02d}] Starting inference", flush=True)

		try:
			stdout_data, stderr_data, memory = run_single_example(command)
		except Exception as exc:  # noqa: BLE001 - we need to persist the failure
			print(f"[Q{question_id:02d}] FAILED: {exc}", flush=True)
			failure_payload = {
				"id": question_id,
				"question": question,
				"error": str(exc),
				"prompt": prompt,
			}
			raw_outputs.append(failure_payload)
			parsed_outputs.append(failure_payload)
			continue

		trimmed_stdout = trim_stdout_to_end_token(stdout_data)
		perf_lines = extract_perf_lines(stderr_data)
		memory_stats = summarize_memory_samples(memory.get("memory_samples", []))
		pretty_print_block(question_id, "STDOUT", stdout_data)
		pretty_print_block(question_id, "STDERR (perf)", "\n".join(perf_lines))
		print(
			f"[Q{question_id:02d}] Completed | answer chars={len(trimmed_stdout)} | perf_lines={len(perf_lines)}",
			flush=True,
		)

		raw_item = {
			"id": question_id,
			"question": question,
			"gold_answers": gold_answers,
			"prompt": prompt,
			"model_answer": stdout_data,
			"stderr": "\n".join(perf_lines),
			**memory,
		}
		raw_outputs.append(raw_item)

		base_parsed = {k: v for k, v in raw_item.items() if k != "memory_samples"}
		base_parsed["model_answer"] = trimmed_stdout
		parsed_item = {
			**base_parsed,
			**memory_stats,
			**parse_metrics(perf_lines),
		}
		parsed_outputs.append(parsed_item)

		raw_metrics_payload["entries"].append(
			{
				"id": question_id,
				"stderr_lines": perf_lines,
			}
		)

	raw_file = output_dir / f"raw_outputs_{model_slug}.json"
	parsed_file = output_dir / f"parsed_outputs_{model_slug}.json"
	metrics_file = output_dir / "raw_metrics.json"

	with raw_file.open("w", encoding="utf-8") as handle:
		json.dump(raw_outputs, handle, indent=2, ensure_ascii=False)

	with parsed_file.open("w", encoding="utf-8") as handle:
		json.dump(parsed_outputs, handle, indent=2, ensure_ascii=False)

	with metrics_file.open("w", encoding="utf-8") as handle:
		json.dump(raw_metrics_payload, handle, indent=2, ensure_ascii=False)

	print(f"Saved raw outputs to {raw_file}")
	print(f"Saved parsed outputs to {parsed_file}")
	print(f"Saved raw metrics to {metrics_file}")


if __name__ == "__main__":
	main()

