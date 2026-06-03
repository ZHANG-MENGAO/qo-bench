# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
# aggqa/run.py
"""Main loop: 302 questions -> retrieve -> prompt -> LLM -> parse -> predictions.json.

Async + concurrent (semaphore-bounded LLM calls) + resumable (skips qids
already in predictions.json) + incremental write (each completed question
is flushed to disk immediately).

Usage:
  python -m aggqa.run --smoke 5            # First N questions only
  python -m aggqa.run                      # All 302, default concurrency
  python -m aggqa.run --concurrency 8      # Adjust parallelism
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from aggqa import config
from aggqa.infra.benchmark_loader import load_benchmark
from aggqa.infra.embedding import embed_query
from aggqa.infra.event_definitions import with_event_definitions
from aggqa.infra.question_params import extract_retriever_scope
from aggqa.infra.retriever import build_retriever

# Import handover modules — they live in a sibling dir; add to path
from aggqa.prompts.prompt_template import SYSTEM_PROMPT, parse_model_output  # noqa: E402

from aggqa.infra.user_prompt import build_user_message  # noqa: E402

# v2 bundle requires the SEC-grounded event_definitions.md be pasted into
# the system prompt of every paradigm (eval_bundle_v2_2026-05-14/README.md).
FULL_SYSTEM_PROMPT = with_event_definitions(SYSTEM_PROMPT)

from aggqa.infra.question_renderer import render_question
from aggqa.infra.structured_output import build_response_format


DEFAULT_CONCURRENCY = 5


def make_llm() -> ChatOpenAI:
    """Build the chat LLM. Two backends:
      - VLLM_CHAT_BASE_URL set  → local vLLM (the cluster PBS path; api_key="EMPTY")
      - otherwise               → OpenRouter (laptop dev path)
    Presence of the env var is the switch. Mirrors lc_oracle_run.py.
    """
    vllm_url = os.environ.get("VLLM_CHAT_BASE_URL")
    if vllm_url:
        return ChatOpenAI(
            base_url=vllm_url,
            api_key="EMPTY",
            model=os.environ.get("VLLM_CHAT_MODEL_NAME", "qwen-3.6-27b"),
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=config.LLM_RETRY_ATTEMPTS,
        )
    return ChatOpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.load_openrouter_key(),
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT_S,
        max_retries=config.LLM_RETRY_ATTEMPTS,
    )


def make_llm_for_question(llm, template_id: str):
    """Return an llm with response_format bound for this question's template."""
    rf = build_response_format(template_id)
    if rf is None:
        return llm
    return llm.bind(extra_body={"response_format": rf})


def parse_answer_for_template(template_id: str, raw_text: str) -> list:
    """Per-template parsing. Handles three model output shapes:

    1. List[Dict] — standard event-tuple shape (handover's parse_model_output).
    2. List[str]  — Entity-projection templates where the LLM follows the
       question's "Return distinct firm tickers" instruction literally and
       emits e.g. `["BECN", "HAIN"]`. the upstream handover parser drops these
       (its final filter is `[t for t in parsed if isinstance(t, dict)]`),
       which silently zeros out recall on A.1.2 / A.3.1 (T18 form) / B.3.1 /
       B.3.2 / B.3.3 / B.4.1 / B.4.2 whenever the LLM picks the string form.
       Verified on v3 1pt smoke 2026-05-21 (B.3.1: LLM returned
       `["BECN","HAIN"]` matching 2/4 GT firms, parsed to []). We promote
       each string to `{"firm_ticker": <s>}` so list_recall's identity-key
       match (firm_ticker) sees the actual answer.
    3. Mixed list — keep dicts as-is, promote strings to firm_ticker dicts.
    """
    parsed_dicts = parse_model_output(raw_text)
    if parsed_dicts:
        return parsed_dicts

    # Fallback: try List[str] / mixed shape.
    if raw_text is None:
        return []
    s = str(raw_text).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    fb = s.find("[")
    lb = s.rfind("]")
    if fb < 0 or lb <= fb:
        return []
    try:
        parsed = json.loads(s[fb : lb + 1])
    except Exception:
        return []
    if not isinstance(parsed, list) or not parsed:
        return []
    out: list = []
    for x in parsed:
        if isinstance(x, dict):
            out.append(x)
        elif isinstance(x, str) and x.strip():
            out.append({"firm_ticker": x.strip()})
    return out


def format_jsonl_record(qid: str, answer) -> str:
    """Serialize one prediction record as a single JSONL line."""
    return json.dumps({"qid": qid, "answer": answer}, ensure_ascii=False) + "\n"


def doc_to_chunk_dict(d):
    """Convert LangChain Document (from MilvusHybridRetriever) -> handover format_chunk input.

    The handover format_chunk emits `[{idx}] {dt} | {tk} | {title}\\n{body}` and
    has no URL slot. Because the templates_config schema requires `cited_urls`,
    the model would otherwise be forced to hallucinate. We piggyback the real
    article_url onto the title (`<url: ...>` suffix) so the LLM has a concrete
    citation to copy. Keeps the handover module unmodified.
    """
    date_int = d.metadata.get("date", 0)
    if isinstance(date_int, int) and date_int > 19000000:
        y, m, day = date_int // 10000, (date_int // 100) % 100, date_int % 100
        date_str = f"{y:04d}-{m:02d}-{day:02d}"
    else:
        date_str = str(date_int)
    tickers = d.metadata.get("tickers") or []
    # F2: show every ticker so multi-firm digest articles don't appear
    # single-firm in the prompt header.
    ticker = ", ".join(tickers) if isinstance(tickers, list) and tickers else "?"
    title = d.metadata.get("title", "")
    url = d.metadata.get("article_url", "")
    if url:
        title = f"{title} <url: {url}>"
    return {
        "body": d.page_content,
        "title": title,
        "date": date_str,
        "ticker": ticker,
    }


def _retrieval_query(question_text: str) -> str:
    # F1: only the first paragraph (the human-language ask) is sent to
    # BM25/embedding. The JSON-schema tail is for the LLM, not retrieval —
    # including it dilutes signal terms ("APA"/"M&A_rumor") with schema words
    # ("string"/"required"/"JSON") in the BM25 ranking.
    return question_text.split("\n\n", 1)[0].strip()


def _retrieve_sync(retriever, q):
    """Synchronous retrieval. Caller MUST hold retriever_lock around this
    because retriever.current_w_start / current_w_end / current_ticker are
    mutable state."""
    scope = extract_retriever_scope(q)
    retriever.current_w_start = scope["w_start"]
    retriever.current_w_end = scope["w_end"]
    # F3: ticker pre-filter; reset to None for templates without F /
    # F_anchor so the previous question's ticker doesn't leak across calls.
    retriever.current_ticker = scope["ticker"]
    return retriever.invoke(_retrieval_query(render_question(q)))


def _load_retrieval_cache(path) -> dict[str, list]:
    """Read a precache JSONL[.gz] and return {qid -> list[Document]}.

    Cache schema (one line per qid):
      {"qid": str, "template_id": str, "scope": {...},
       "docs": [{"page_content": str, "metadata": {...}}, ...],
       "retrieval_meta": {"top_k_reranked": int, "rrf_k": int, ...}}

    Reads .jsonl and .jsonl.gz. Validates that retrieval_meta.top_k_reranked
    matches config.TOP_K_RERANKED on the FIRST line — fails fast if the cache
    was built with a different K than the runner expects.
    """
    p = Path(path)
    opener = gzip.open if p.name.endswith(".gz") else open
    cache: dict[str, list] = {}
    first_line_checked = False
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec:
                # Precache wrote an error-line for this qid. Skip — the runner
                # will treat the qid as cache-miss and raise a clear KeyError
                # naming the qid, which is preferable to silently using zero docs.
                continue
            if not first_line_checked:
                meta = rec.get("retrieval_meta", {})
                cache_k = meta.get("top_k_reranked")
                if cache_k is not None and cache_k != config.TOP_K_RERANKED:
                    raise ValueError(
                        f"Cache top_k_reranked={cache_k} != config.TOP_K_RERANKED="
                        f"{config.TOP_K_RERANKED}. Rebuild the cache, or revert "
                        f"config to match. Cache file: {p}"
                    )
                first_line_checked = True
            docs = [
                Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                for d in rec["docs"]
            ]
            cache[rec["qid"]] = docs
    return cache


async def process_question(q, retriever, retriever_lock, llm, llm_sem, retrieval_cache=None):
    qid = q["qid"]
    template_id = q["id"]
    try:
        if retrieval_cache is not None:
            if qid not in retrieval_cache:
                raise KeyError(f"qid {qid!r} not in retrieval cache — rebuild or check cache file")
            docs = retrieval_cache[qid]
        else:
            async with retriever_lock:
                docs = await asyncio.to_thread(_retrieve_sync, retriever, q)
        chunks_for_prompt = [doc_to_chunk_dict(d) for d in docs]
        # F4: override the handover default of 1500 chars. the maintainer confirmed
        # 2026-05-08 ("chunk 有多少就塞多少进 prompt，没有理由截断"). Milvus
        # caps chunk_text at 4000 chars (CHUNK_SIZE), so 4000 is effectively
        # "no truncation".
        user_msg = build_user_message(q, chunks_for_prompt, max_body_chars=4000)
        bound_llm = make_llm_for_question(llm, template_id)
        async with llm_sem:
            ai_msg = await bound_llm.ainvoke([
                {"role": "system", "content": FULL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
        raw = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
        answer = parse_answer_for_template(template_id, raw)
        log_record = {
            "qid": qid,
            "template_id": template_id,
            "n_chunks": len(docs),
            "url_hashes": [d.metadata.get("url_hash") for d in docs],
            "raw": raw,
            "parsed_n_tuples": len(answer) if isinstance(answer, list) else 1,
            "response_metadata": dict(ai_msg.response_metadata or {}),
        }
    except KeyError:
        raise   # cache-miss must propagate, not be swallowed
    except Exception as e:
        # Preserve full traceback — see memory: feedback_preserve_full_traceback.
        answer = 0 if template_id == "B.1.5" else []   # B.1.5 is the composite_b16 (scalar int) template
        log_record = {
            "qid": qid,
            "template_id": template_id,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
    return qid, {"qid": qid, "answer": answer}, log_record


def _load_existing_predictions() -> tuple[list[dict], set[str]]:
    if not config.PREDICTIONS_PATH.exists():
        return [], set()
    preds: list[dict] = []
    done: set[str] = set()
    try:
        with config.PREDICTIONS_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                preds.append(rec)
                if "qid" in rec:
                    done.add(rec["qid"])
        return preds, done
    except Exception as e:
        print(f"Warning: existing predictions.jsonl unreadable ({e}); starting fresh")
        return [], set()


async def run_async(
    smoke: int | None,
    concurrency: int,
    benchmark_path=None,
    per_template_cap: int | None = None,
    retrieval_cache_path=None,
) -> None:
    bench = benchmark_path or config.BENCHMARK_PATH
    questions = load_benchmark(bench, per_template_cap=per_template_cap)
    print(f"Benchmark: {bench}")
    if per_template_cap is not None:
        print(f"per-template cap: {per_template_cap} → {len(questions)} questions total")
    if smoke:
        questions = questions[:smoke]
        print(f"SMOKE mode: first {smoke} questions only")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions, done_qids = _load_existing_predictions()
    pending = [q for q in questions if q["qid"] not in done_qids]

    print(f"Total target questions: {len(questions)}")
    print(f"Already in predictions.jsonl: {len(done_qids)} (skipped)")
    print(f"To process: {len(pending)} (concurrency={concurrency})")
    if not pending:
        print("Nothing to do.")
        return

    if retrieval_cache_path is not None:
        print(f"Retrieval cache mode: loading {retrieval_cache_path}")
        retrieval_cache = _load_retrieval_cache(retrieval_cache_path)
        print(f"  cache size: {len(retrieval_cache)} qids")
        retriever = None
    else:
        retrieval_cache = None
        retriever = build_retriever(embed_query)

    llm = make_llm()
    retriever_lock = asyncio.Lock()
    llm_sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    log_file = config.RUN_LOG_PATH.open("a")  # append, not overwrite — preserves prior runs
    completed = len(done_qids)
    total = len(questions)
    t0 = time.time()

    async def process_and_write(q):
        nonlocal completed
        qid, prediction, log = await process_question(
            q, retriever, retriever_lock, llm, llm_sem,
            retrieval_cache=retrieval_cache,
        )
        async with write_lock:
            predictions.append(prediction)
            # JSONL append: one line per completed question. Each line is
            # self-contained, so a crashed run still leaves a valid file.
            line = format_jsonl_record(prediction["qid"], prediction["answer"])
            with config.PREDICTIONS_PATH.open("a") as f:
                f.write(line)
            log_file.write(json.dumps(log) + "\n")
            log_file.flush()
            completed += 1
            elapsed = time.time() - t0
            ntup = log.get("parsed_n_tuples", "ERR")
            print(
                f"[{completed}/{total}] {qid} ({log.get('template_id')}) "
                f"parsed={ntup}  elapsed={elapsed:.0f}s",
                flush=True,
            )

    tasks = [asyncio.create_task(process_and_write(q)) for q in pending]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    log_file.close()

    n_exc = sum(1 for r in results if isinstance(r, Exception))
    elapsed = time.time() - t0
    print(f"\nAll done. {completed} predictions in predictions.jsonl. "
          f"Wall: {elapsed:.0f}s. Task exceptions: {n_exc}")
    print(f"  -> {config.PREDICTIONS_PATH}")
    print(f"  -> {config.RUN_LOG_PATH}")


def _redirect_run_outputs(name: str) -> None:
    """Point predictions/results/log/notes at a sibling dir of outputs/.
    SCHEMA_PATH and RUNTIME_PATH stay in canonical outputs/ — they are
    pre-flight discovery artifacts, not run-specific."""
    new_dir = config.OUTPUTS_DIR.parent / name
    new_dir.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR = new_dir
    config.PREDICTIONS_PATH = new_dir / "predictions.jsonl"
    config.RESULTS_PATH = new_dir / "results.json"
    config.RUN_LOG_PATH = new_dir / "run_log.txt"
    config.RUN_NOTES_PATH = new_dir / "RUN_NOTES.md"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", type=int, default=None, help="Run only first N questions")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="Max concurrent LLM calls (default 5)")
    p.add_argument("--output-dir", default=None,
                   help="Sibling dir name for predictions/results/log "
                        "(default: outputs/). Use to keep multiple runs side-by-side.")
    p.add_argument("--benchmark", default=None,
                   help="Path to benchmark JSON/JSONL. Defaults to config.BENCHMARK_PATH.")
    p.add_argument("--per-template-cap", type=int, default=None,
                   help="Cap questions per template_id (e.g. --per-template-cap 1 "
                        "with the cap50 alive set = 21-q smoke). Applied BEFORE --smoke.")
    p.add_argument("--retrieval-cache", default=None,
                   help="Path to precomputed retrieval cache (JSONL or .jsonl.gz). "
                        "When set, skip Milvus and read docs from cache by qid. "
                        "Required when running on a host that can't reach Milvus (e.g. the cluster).")
    args = p.parse_args()
    if args.output_dir:
        _redirect_run_outputs(args.output_dir)
    bench_path = None
    if args.benchmark:
        bp = Path(args.benchmark)
        bench_path = bp if bp.is_absolute() else (config.PROJECT_ROOT / bp)
    cache_path = None
    if args.retrieval_cache:
        cp = Path(args.retrieval_cache)
        cache_path = cp if cp.is_absolute() else (config.PROJECT_ROOT / cp)
    asyncio.run(run_async(
        smoke=args.smoke, concurrency=args.concurrency,
        benchmark_path=bench_path, per_template_cap=args.per_template_cap,
        retrieval_cache_path=cache_path,
    ))


if __name__ == "__main__":
    main()
