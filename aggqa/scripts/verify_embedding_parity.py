# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Verify OpenRouter Qwen3-Embedding-4B parity against local Qwen3-Embedding-4B.

Calls both backends directly (bypassing env-var dispatch) on the same sample
of real benchmark queries, computes per-query cosine similarity, and reports
median / min / max. Passes if **median >= 0.995 and min >= 0.99** — a tighter
bar than typical retrieval drift but loose enough to tolerate fp32 vs bf16
serving precision (which empirically lands at ~0.9995-0.9999 for same-model).

Why this matters: corpus was indexed using Qwen3-Embedding-4B on the upstream
SPC server. We don't know SPC's exact precision, but our local fp32 setup is
the closest reproduction we control. If OpenRouter's served version differs
materially from local fp32, top-k retrieval will silently drift and every
paper number computed via OpenRouter will be off.

Usage:
    python -m aggqa.scripts.verify_embedding_parity
    python -m aggqa.scripts.verify_embedding_parity --n 10
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

from aggqa import config
from aggqa.infra import embedding as _emb


def render_query_text(q: dict) -> str:
    """Use the benchmark's nl_question as the query text — same input the
    runner would actually feed embed_query in practice."""
    return q["nl_question"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of queries to test")
    ap.add_argument("--bench", type=Path, default=config.BENCHMARK_PATH)
    ap.add_argument("--median-min", type=float, default=0.995)
    ap.add_argument("--abs-min", type=float, default=0.99)
    args = ap.parse_args()

    # Load N queries, evenly spread across templates so we don't pick all
    # from one template's flavor.
    with gzip.open(args.bench, "rt") as f:
        all_qs = [json.loads(l) for l in f if l.strip()]
    by_template: dict[str, list[dict]] = {}
    for q in all_qs:
        by_template.setdefault(q["id"], []).append(q)
    # Round-robin across templates, take N
    sampled = []
    tids = sorted(by_template.keys())
    for i in range(args.n):
        tid = tids[i % len(tids)]
        sampled.append(by_template[tid][i // len(tids)])

    print(f"Benchmark: {args.bench}")
    print(f"Testing {len(sampled)} queries across {len(set(q['id'] for q in sampled))} templates")
    print(f"Pass bar: median cos >= {args.median_min}, min cos >= {args.abs_min}")
    print()

    cos_sims: list[float] = []
    for i, q in enumerate(sampled, 1):
        text = render_query_text(q)
        t0 = time.time()
        local_vec = _emb._embed_local(text)
        t_local = time.time() - t0

        t0 = time.time()
        api_vec = _emb._embed_deepinfra(text)
        t_api = time.time() - t0

        cos = cosine(local_vec, api_vec)
        cos_sims.append(cos)
        verdict = "✓" if cos >= args.abs_min else "✗"
        print(f"[{i}/{len(sampled)}] {q['id']:14s}  cos={cos:.6f}  "
              f"local={t_local:5.2f}s  api={t_api:5.2f}s  {verdict}")
        # Brief preview of the query text so user can sanity-check input
        print(f"          q: {text[:100]}{'...' if len(text) > 100 else ''}")

    cos_sims.sort()
    median = cos_sims[len(cos_sims) // 2]
    mn, mx = cos_sims[0], cos_sims[-1]
    print()
    print(f"Summary:  median={median:.6f}  min={mn:.6f}  max={mx:.6f}")

    passed = (median >= args.median_min) and (mn >= args.abs_min)
    if passed:
        print(f"\nPASS — OK to switch to AGG_QA_EMBED_BACKEND=deepinfra.")
        return 0
    else:
        print(f"\nFAIL — drift exceeds bar. Inspect manually before swapping backends.")
        print(f"  median {median:.6f} {'<' if median < args.median_min else '>='} {args.median_min} (bar)")
        print(f"  min    {mn:.6f} {'<' if mn < args.abs_min else '>='} {args.abs_min} (bar)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
