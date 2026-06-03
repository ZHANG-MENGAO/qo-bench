# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""No-context (closed-book) baseline runner — same model, same questions,
same eval pipeline as LC-oracle, but NO articles in the user message. The
LLM answers from its parametric knowledge alone; this measures the
knowledge-floor companion to LC-oracle's capability ceiling.

Spec: docs/superpowers/specs/2026-05-22-no-context-baseline-design.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from langchain_openai import ChatOpenAI

from aggqa import config

# Belt-and-suspenders: both lc_oracle_run (imported below) and run.py insert
# HANDOVER_DIR, but we do it here explicitly so import-order changes don't
# silently break prompt_template availability.

from aggqa.infra.event_definitions import with_event_definitions
from aggqa.infra.output_schemas import get_schema_for_template
from aggqa.infra.question_renderer import render_question
from aggqa.baselines.lc_oracle import (
    _append_jsonl,
    _iter_questions,
    _load_existing_errors,
    _load_existing_predictions,
    make_llm,
)
from aggqa.baselines.naive_rag import parse_answer_for_template


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NO_CONTEXT_SYSTEM_PROMPT_BASE = """\
You answer questions about corporate events using your prior knowledge of \
companies, deals, executives, splits, and filings. No articles are provided \
in this run — answer from what you know. Recall and list any events from \
your training that may match the question criteria, even if you are not \
fully certain about exact dates or details; partial knowledge is useful.

You output a strict JSON array matching the schema given in the question. \
No commentary, no markdown fences, no prose, no prefix, no suffix. Just the \
JSON array starting with [ and ending with ]. If genuinely no events come \
to mind for this firm and window, output [].\
"""

FULL_SYSTEM_PROMPT = with_event_definitions(NO_CONTEXT_SYSTEM_PROMPT_BASE)
MODEL_NAME = config.LLM_MODEL


# ---------------------------------------------------------------------------
# User message builder (no `# Articles` section — see spec §Architecture)
# ---------------------------------------------------------------------------

def build_no_context_user_message(q: dict) -> str:
    schema_block = get_schema_for_template(q["id"], q.get("legacy_id"))
    nl = render_question(q)
    return f"{schema_block}\n# Question\n\n{nl}"


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

async def process_question(
    q: dict,
    *,
    llm: ChatOpenAI,
    sem: asyncio.Semaphore,
    model_name: str = MODEL_NAME,
) -> dict:
    qid = q["qid"]
    template_id = q.get("id", "?")

    user_msg = build_no_context_user_message(q)

    t0 = time.perf_counter()
    async with sem:
        ai_msg = await llm.ainvoke([
            {"role": "system", "content": FULL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    elapsed = time.perf_counter() - t0

    raw = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    answer = parse_answer_for_template(template_id, raw)

    usage = getattr(ai_msg, "usage_metadata", {}) or {}
    additional = getattr(ai_msg, "additional_kwargs", {}) or {}
    response_md = getattr(ai_msg, "response_metadata", {}) or {}
    reasoning_content = additional.get("reasoning") or response_md.get("reasoning")
    raw_token_usage = (response_md.get("token_usage") or {})
    completion_details = raw_token_usage.get("completion_tokens_details") or {}
    reasoning_tokens = (
        usage.get("reasoning_tokens")
        or completion_details.get("reasoning_tokens")
        or (usage.get("output_token_details") or {}).get("reasoning")
    )
    return {
        "qid": qid,
        "template_id": template_id,
        "answer": answer,
        "raw_output": raw,
        "reasoning_content": reasoning_content,
        "n_chunks_input": 0,
        "n_chunks_kept": 0,
        "n_chunks_truncated": 0,
        "truncated": False,
        "per_event_min_1": False,
        "input_tokens_estimate": 0,
        "input_tokens_actual": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "model": model_name,
        "slice_source": "no_context",
        "oversize": False,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_async(
    *,
    benchmark: Path,
    output_dir: Path,
    concurrency: int,
    backend: str,
    smoke: int | None = None,
    model: str | None = None,
    reasoning_effort: str = "low",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"

    _, done_qids = _load_existing_predictions(output_dir)
    err_qids = _load_existing_errors(output_dir)
    skip = done_qids | err_qids

    questions = [q for q in _iter_questions(Path(benchmark)) if q["qid"] not in skip]
    if smoke is not None:
        questions = questions[:smoke]

    llm = make_llm(backend, model=model, reasoning_effort=reasoning_effort)

    # Thinking is ON (Qwen3 chat-template default). Job 488832 ran with
    # thinking=off and produced 100% [] answers — the system prompt's old
    # "if uncertain, output []" clause combined with single-step decoding
    # made the model take the trivial path. We removed that clause from the
    # prompt to encourage actual recall, but recall requires thinking time;
    # without <think> the model still can't deliberate over candidates.
    #
    # Cap max_tokens to prevent the spiral observed on 488528 (where some
    # questions ate all 61120 tokens in thinking-then-empty before timing
    # out). 16384 covers the natural distribution of thinking length seen
    # in 488528's non-empty answers (median 5K, p95 ~12K, max 12.1K).
    llm.max_tokens = 16384
    effective_model = model or MODEL_NAME
    print(f"[no_context] backend={backend}  model={effective_model}  "
          f"reasoning_effort={reasoning_effort}  concurrency={concurrency}  "
          f"thinking=on  max_tokens={llm.max_tokens}", flush=True)
    sem = asyncio.Semaphore(concurrency)

    shutdown = asyncio.Event()
    def _request_shutdown(signame):
        if not shutdown.is_set():
            print(f"[no_context] {signame} received — finalize after current question", flush=True)
            shutdown.set()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except (NotImplementedError, RuntimeError):
            pass   # pytest / Windows

    n_done = 0
    n_error = 0
    write_lock = asyncio.Lock()

    async def _process_and_write(q):
        nonlocal n_done, n_error
        if shutdown.is_set():
            return
        qid = q["qid"]
        try:
            pred = await process_question(
                q, llm=llm, sem=sem, model_name=effective_model,
            )
            async with write_lock:
                _append_jsonl(preds_path, pred)
                n_done += 1
        except Exception as e:
            async with write_lock:
                _append_jsonl(errors_path, {
                    "qid": qid,
                    "template_id": q.get("id", "?"),
                    "reason": "exception",
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                })
                n_error += 1
            print(f"[no_context] {qid}: {type(e).__name__}: {e}", flush=True)

    tasks = [asyncio.create_task(_process_and_write(q)) for q in questions]
    await asyncio.gather(*tasks, return_exceptions=True)

    print(f"[no_context] done={n_done}  errors={n_error}", flush=True)


def run_sync(**kwargs) -> None:
    asyncio.run(run_async(**kwargs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="No-context (closed-book) baseline runner")
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Path to benchmark/questions/questions.jsonl.gz")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("aggqa/runs/outputs_no_context"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--smoke", type=int, default=None,
                        help="Run only first N questions (for local smoke)")
    parser.add_argument("--backend", choices=["openrouter", "vllm"],
                        default=os.environ.get("LC_ORACLE_LLM_BACKEND", "openrouter"))
    parser.add_argument("--model", type=str, default=None,
                        help="Override config.LLM_MODEL (openrouter only)")
    parser.add_argument("--reasoning-effort",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        default="low")
    args = parser.parse_args()

    run_sync(
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        backend=args.backend,
        smoke=args.smoke,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
