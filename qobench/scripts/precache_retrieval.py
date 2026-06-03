# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Phase-1 driver: pre-retrieve top-K chunks for every question in a benchmark
and write a JSONL cache that run.py can replay later.

Why: compute nodes may not be able to reach the Milvus host directly. Run this
where Milvus is reachable, copy the resulting cache to the compute environment,
and replay it via run.py --retrieval-cache.

Resumable: appends one line per qid; reruns skip qids already in the output.

Usage:
    QOBENCH_EMBED_BACKEND=deepinfra \\
    DEEPINFRA_API_KEY=... \\
    MILVUS_TOKEN=root:... \\
    python -m qobench.scripts.precache_retrieval \\
        --benchmark benchmark/questions/questions.jsonl.gz \\
        --output    retrieval_cache.jsonl \\
        --concurrency 8

After completion, gzip the output: `gzip retrieval_cache.jsonl`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

from pymilvus import MilvusClient

from qobench import config
from qobench.infra.benchmark_loader import load_benchmark
from qobench.infra.embedding import embed_query
from qobench.infra.question_params import extract_retriever_scope
from qobench.infra.question_renderer import render_question
from qobench.infra.retriever import MilvusHybridRetriever
from qobench.infra.whitelist import load_v2_whitelist


def _retrieval_query(question_text: str) -> str:
    """F1 trim: only the first paragraph (human-language ask) reaches BM25."""
    return question_text.split("\n\n", 1)[0].strip()


def _make_shared_client() -> MilvusClient:
    """One pymilvus client per process (thread-safe per pymilvus docs)."""
    return MilvusClient(
        uri=config.MILVUS_URI,
        token=config.MILVUS_TOKEN,
        db_name=config.MILVUS_DB_NAME,
    )


def _make_per_question_retriever(q, shared_client, whitelist):
    """Build a FRESH MilvusHybridRetriever per question, wrapping the shared
    MilvusClient. Mirrors react_agent.make_question_scoped_retriever but uses
    Naive-RAG K (100 → 30, not REACT's 50 → 10).

    Two reasons for per-question instances:
    1. Concurrency: shared-instance mutation requires a serializing lock,
       collapsing --concurrency=N to effectively N=1.
    2. Correctness: seen_url_hashes accumulates per retriever instance. A
       shared retriever leaks cross-round dedup state from question N into
       question N+1's filter — Naive RAG must not dedup across questions.
    """
    rt = config.load_runtime()
    r = MilvusHybridRetriever(
        client=shared_client,
        collection_name=rt["collection_name"],
        vector_field=rt["vector_field"],
        sparse_field=rt.get("sparse_field"),
        date_field=rt["date_field"],
        date_format=rt["date_format"],
        url_hash_field=rt["url_hash_field"],
        embedding_fn=embed_query,
        rerank_api_key=config.load_deepinfra_key(),
        url_hash_whitelist=whitelist,
        top_k_initial=config.TOP_K_INITIAL,
        top_k_reranked=config.TOP_K_RERANKED,
    )
    scope = extract_retriever_scope(q)
    r.current_w_start = scope["w_start"]
    r.current_w_end = scope["w_end"]
    r.current_ticker = scope["ticker"]
    return r, scope


def _retrieve_sync(q, shared_client, whitelist):
    """One question end-to-end: build retriever, run hybrid+rerank, return
    (docs, scope). Safe to call from many threads concurrently because the
    retriever is freshly built per call."""
    r, scope = _make_per_question_retriever(q, shared_client, whitelist)
    docs = r.invoke(_retrieval_query(render_question(q)))
    return docs, scope


def _json_safe(v):
    """Coerce pymilvus protobuf containers (e.g. RepeatedScalarContainer for
    ARRAY fields like `tickers`) into plain Python types. Without this,
    json.dumps raises TypeError on every retrieved doc that has a non-empty
    array column. Idempotent for already-safe values."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, dict):
        return {k: _json_safe(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    try:
        return [_json_safe(x) for x in v]  # iterable protobuf container
    except TypeError:
        return str(v)


def _serialize_one(q, docs, scope) -> str:
    """JSONL line per qid. Schema matches spec §4."""
    return json.dumps({
        "qid": q["qid"],
        "template_id": q["id"],
        "scope": scope,
        "docs": [
            {"page_content": d.page_content,
             "metadata": _json_safe(dict(d.metadata or {}))}
            for d in docs
        ],
        "retrieval_meta": {
            "embed_backend":  os.environ.get("QOBENCH_EMBED_BACKEND", "local"),
            "rerank_backend": "deepinfra",
            "top_k_initial":  config.TOP_K_INITIAL,
            "top_k_reranked": config.TOP_K_RERANKED,
            "rrf_k":          config.RRF_K,
            "date_pad_days":  config.DATE_WINDOW_PAD_DAYS,
            "whitelist":      "load_v2_whitelist (canonical 22,984-article substrate, v2↔v3 identical)",
            "retriever_version": "2026-05-22-precache",
        },
    }, ensure_ascii=False) + "\n"


def _load_done_qids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "error" in rec:
                # Error-line: don't count as done; resume will retry this qid.
                continue
            qid = rec.get("qid")
            if qid:
                done.add(qid)
    return done


async def run_async(benchmark_path: Path, output_path: Path, concurrency: int) -> None:
    questions = load_benchmark(benchmark_path, per_template_cap=None)
    done = _load_done_qids(output_path)
    pending = [q for q in questions if q["qid"] not in done]
    print(f"Benchmark : {benchmark_path}")
    print(f"Total q   : {len(questions)}")
    print(f"Already   : {len(done)}")
    print(f"To process: {len(pending)} (concurrency={concurrency})")
    if not pending:
        print("Nothing to do.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shared_client = _make_shared_client()
    whitelist     = load_v2_whitelist()  # ~22,984 url_hashes, load once
    sem           = asyncio.Semaphore(concurrency)
    write_lock    = asyncio.Lock()

    completed = 0
    t0 = time.time()

    async def process(q):
        nonlocal completed
        try:
            async with sem:
                docs, scope = await asyncio.to_thread(
                    _retrieve_sync, q, shared_client, whitelist
                )
            line = _serialize_one(q, docs, scope)
        except Exception as e:
            line = json.dumps({
                "qid": q["qid"], "template_id": q["id"],
                "error": repr(e), "traceback": traceback.format_exc(),
            }, ensure_ascii=False) + "\n"
        async with write_lock:
            with output_path.open("a") as f:
                f.write(line)
            completed += 1
            elapsed = time.time() - t0
            print(f"[{completed}/{len(pending)}] {q['qid']} elapsed={elapsed:.0f}s", flush=True)

    tasks = [asyncio.create_task(process(q)) for q in pending]
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Done. {completed} new qids. Wall: {time.time() - t0:.0f}s. → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, help="Path to benchmark .json/.jsonl/.jsonl.gz")
    p.add_argument("--output",    required=True, help="Output JSONL path (append-only, resumable)")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    bench  = Path(args.benchmark)
    output = Path(args.output)
    if not bench.is_absolute():
        bench = config.PROJECT_ROOT / bench
    asyncio.run(run_async(bench, output, args.concurrency))


if __name__ == "__main__":
    main()
