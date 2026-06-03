# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""LC-oracle baseline runner — feed only golden chunks to a
single Qwen3.6-27B call at 262K native context. Capability-ceiling reference for
the paper; mirrors run.py's resumable+atomic-write+async-semaphore pattern.

Spec: docs/superpowers/specs/2026-05-18-lc-oracle-baseline-design.md
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_openai import ChatOpenAI

from aggqa import config

# Make the handover prompt_template importable (skill convention from run.py)
from aggqa.prompts.prompt_template import (  # noqa: E402
    SYSTEM_PROMPT,
    parse_model_output,  # noqa: F401  (imported for sys.path side-effect / availability)
)

from aggqa.infra.articles_index import load_articles_index
from aggqa.infra.event_definitions import with_event_definitions
from aggqa.infra.golden_chunks import (
    BundleSkewError,
    OversizeQuestionError,
    build_oracle_context,
    build_oracle_context_budget_aware,
    estimate_input_tokens,
)
from aggqa.infra.user_prompt import build_user_message
from aggqa.baselines.naive_rag import parse_answer_for_template


FULL_SYSTEM_PROMPT = with_event_definitions(SYSTEM_PROMPT)
MODEL_NAME = config.LLM_MODEL   # "qwen/qwen3.6-27b" by default


# ---------------------------------------------------------------------------
# LLM factory — OpenRouter (smoke) vs vLLM (the cluster)
# ---------------------------------------------------------------------------

def make_llm(backend: str, model: str | None = None,
             reasoning_effort: str = "low") -> ChatOpenAI:
    """Build a ChatOpenAI client for either backend.

    `model` overrides config.LLM_MODEL — used by --model CLI flag so we can
    swap baseline model (e.g. deepseek/deepseek-v4-flash) without editing
    config. For vllm backend, model name is fixed to whatever vllm is serving
    (the --served-model-name in the PBS script), so this override only
    affects the openrouter path.

    `reasoning_effort` is one of "low" / "medium" / "high" — openrouter only.
    Different models interpret these levels differently (e.g. v4-flash treats
    "low" as "do not think at all"; r1 always thinks regardless).
    """
    max_out = config.LC_ORACLE_OUTPUT_RESERVE - 1024
    if backend == "vllm":
        base_url = os.environ["VLLM_CHAT_BASE_URL"]
        return ChatOpenAI(
            model="qwen-3.6-27b",
            base_url=base_url,
            api_key="EMPTY",
            temperature=config.LLM_TEMPERATURE,
            max_tokens=max_out,
            timeout=config.LLM_TIMEOUT_S,
            # Thinking is enabled (Qwen3 chat-template default). vLLM is
            # served with `--reasoning-parser qwen3`, which routes `<think>`
            # blocks to `reasoning_content` and keeps `content` clean for
            # the JSON answer. Before the reasoning-parser fix we had to
            # set `enable_thinking=False` to keep JSON from being eaten by
            # an unterminated `<think>` block, but that suppressed all
            # reasoning and hurt compositional templates (B.1.2/B.1.3/
            # B.1.5/B.2.1/B.3.2). Leaving the default ON now.
        )
    # OpenRouter (smoke / cross-model baseline)
    # xhigh reasoning on big-context (50K+ input) cap50 questions can take 10+
    # min per call. 1800s timeout = 30 min single-request ceiling; cap50 still
    # finishes in reasonable wall-clock at concurrency=32 since avg is much lower.
    openrouter_timeout = 1800 if reasoning_effort in ("high", "xhigh") else config.LLM_TIMEOUT_S
    return ChatOpenAI(
        model=model or config.LLM_MODEL,
        base_url=config.OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=max_out,
        timeout=openrouter_timeout,
        model_kwargs={"extra_body": {
            "reasoning": {"effort": reasoning_effort},
            "include_reasoning": True,    # exposes reasoning_tokens via output_token_details.reasoning
        }},
    )


# ---------------------------------------------------------------------------
# Resume state — both JSONL
# ---------------------------------------------------------------------------

def _load_existing_predictions(output_dir: Path) -> tuple[list[dict], set[str]]:
    p = output_dir / "predictions.jsonl"
    if not p.exists():
        return [], set()
    preds: list[dict] = []
    done: set[str] = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            preds.append(rec)
            if "qid" in rec:
                done.add(rec["qid"])
        except Exception as e:
            print(f"[lc_oracle] WARN: predictions.jsonl line unreadable: {e}", flush=True)
    return preds, done


def _load_existing_errors(output_dir: Path) -> set[str]:
    p = output_dir / "errors.jsonl"
    if not p.exists():
        return set()
    seen: set[str] = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seen.add(json.loads(line)["qid"])
        except Exception:
            pass
    return seen


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _iter_questions(path: Path) -> Iterable[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

async def process_question(
    q: dict,
    *,
    articles_index: dict,
    articles_df,
    llm: ChatOpenAI,
    sem: asyncio.Semaphore,
    model_name: str = MODEL_NAME,
) -> dict:
    qid = q["qid"]
    template_id = q.get("id", "?")

    # Per-event-min-1 fallback bookkeeping. Set by the budget-aware branch so
    # we can write the degradation footprint into predictions.jsonl.
    truncated = False
    n_chunks_input = None
    n_chunks_dropped = 0

    try:
        chunks = build_oracle_context(q, articles_index)
        slice_source = "golden_chunks"
    except OversizeQuestionError as e:
        # design decision (2026-05-19): rather than DNF the 5 cap50 oversize questions,
        # fall back to per-event-min-1 + round-robin fill so every GT event
        # keeps ≥1 attesting chunk in the input. Trades off strict perfect-
        # oracle semantics for full 768/768 coverage; documented in spec §4.
        chunks, stats = build_oracle_context_budget_aware(q, articles_index)
        truncated = True
        slice_source = "golden_chunks_budget_aware"
        n_chunks_input = stats["n_chunks_input"]
        n_chunks_dropped = stats["n_chunks_dropped"]
        print(
            f"[lc_oracle] {qid} {template_id}: strict oversize "
            f"({e.input_tokens:,} tok / {e.n_chunks} chunks) → budget-aware "
            f"kept {stats['n_chunks_kept']}/{stats['n_chunks_input']} chunks "
            f"({stats['input_tokens_estimate']:,} tok, {stats['n_events']} events)",
            flush=True,
        )

    user_msg = build_user_message(q, chunks, max_body_chars=100_000)

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
    # OpenRouter for DeepSeek puts reasoning content in additional_kwargs.reasoning
    # (when include_reasoning=True) and reasoning_tokens in
    # response_metadata.token_usage.completion_tokens_details.reasoning_tokens.
    # LangChain's normalized usage_metadata sometimes misses these for non-OpenAI
    # providers, so we extract from both paths and take whichever is populated.
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
        "n_chunks_input": n_chunks_input if n_chunks_input is not None else len(chunks),
        "n_chunks_kept": len(chunks),
        "n_chunks_truncated": n_chunks_dropped,
        "truncated": truncated,
        "per_event_min_1": truncated,
        "input_tokens_estimate": estimate_input_tokens(chunks, q),
        "input_tokens_actual": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "model": model_name,
        "slice_source": slice_source,
        "oversize": False,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_async(
    *,
    benchmark: Path,
    articles_csv: Path,
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

    articles_index = load_articles_index(articles_csv)
    articles_df = pd.read_csv(articles_csv, dtype=str).fillna("")

    questions = [q for q in _iter_questions(Path(benchmark)) if q["qid"] not in skip]
    if smoke is not None:
        questions = questions[:smoke]

    llm = make_llm(backend, model=model, reasoning_effort=reasoning_effort)
    effective_model = model or MODEL_NAME
    print(f"[lc_oracle] backend={backend}  model={effective_model}  "
          f"reasoning_effort={reasoning_effort}  concurrency={concurrency}", flush=True)
    sem = asyncio.Semaphore(concurrency)

    # SIGTERM / SIGINT handler — finalize after current question, never mid-write.
    shutdown = asyncio.Event()
    def _request_shutdown(signame):
        if not shutdown.is_set():
            print(f"[lc_oracle] {signame} received — finalize after current question", flush=True)
            shutdown.set()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except (NotImplementedError, RuntimeError):
            pass   # pytest / Windows

    n_done = 0
    n_truncated = 0       # per-event-min-1 fallback applied (strict→budget_aware)
    n_oversize = 0        # defensive: only fires if Pass-1 floor itself > budget
    n_error = 0
    write_lock = asyncio.Lock()

    async def _process_and_write(q):
        nonlocal n_done, n_truncated, n_oversize, n_error
        if shutdown.is_set():
            return
        qid = q["qid"]
        try:
            pred = await process_question(
                q, articles_index=articles_index, articles_df=articles_df,
                llm=llm, sem=sem, model_name=effective_model,
            )
            async with write_lock:
                _append_jsonl(preds_path, pred)
                n_done += 1
                if pred.get("truncated"):
                    n_truncated += 1
        except OversizeQuestionError as e:
            async with write_lock:
                _append_jsonl(errors_path, {
                    "qid": qid,
                    "template_id": q.get("id", "?"),
                    "reason": "oversize_floor",
                    "n_chunks": e.n_chunks,
                    "input_tokens_estimate": e.input_tokens,
                    "input_token_budget": e.budget,
                })
                n_oversize += 1
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
            print(f"[lc_oracle] {qid}: {type(e).__name__}: {e}", flush=True)

    tasks = [asyncio.create_task(_process_and_write(q)) for q in questions]
    await asyncio.gather(*tasks, return_exceptions=True)

    print(
        f"[lc_oracle] done={n_done}  truncated_budget_aware={n_truncated}  "
        f"oversize_floor={n_oversize}  errors={n_error}",
        flush=True,
    )


def run_sync(**kwargs) -> None:
    asyncio.run(run_async(**kwargs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LC-oracle baseline runner")
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Path to benchmark/questions/questions.jsonl.gz")
    parser.add_argument("--articles-csv", type=Path, required=True,
                        help="Path to corpus/articles.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("aggqa/runs/outputs_lc_oracle"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--smoke", type=int, default=None,
                        help="Run only first N questions (for local smoke)")
    parser.add_argument("--backend", choices=["openrouter", "vllm"],
                        default=os.environ.get("LC_ORACLE_LLM_BACKEND", "openrouter"))
    parser.add_argument("--model", type=str, default=None,
                        help="Override config.LLM_MODEL (openrouter only). "
                             "Example: deepseek/deepseek-v4-flash")
    parser.add_argument("--reasoning-effort",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        default="low",
                        help="OpenRouter reasoning effort. Full ladder per "
                             "OpenRouter taxonomy. For DeepSeek v4-pro: "
                             "xhigh = native 'Think Max' mode. Models interpret "
                             "differently (e.g. v4-flash 'low' = no thinking).")
    args = parser.parse_args()

    run_sync(
        benchmark=args.benchmark,
        articles_csv=args.articles_csv,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        backend=args.backend,
        smoke=args.smoke,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
