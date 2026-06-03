# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Qwen3-Embedding-4B query encoder — dual backend (local | DeepInfra).

Backend is selected at import time via env var:
    QOBENCH_EMBED_BACKEND=local      (default — SentenceTransformer on CPU)
    QOBENCH_EMBED_BACKEND=deepinfra  (HTTP /v1/openai/embeddings, no local model)

Why two backends:
- Local fp32 was the original path (the corpus was indexed with this model).
  Loads ~16GB into RAM, encode runs on CPU (~3-4s/query), serialized by an
  encode lock for thread safety. At ReAct concurrency ≥ 5 the lock becomes
  the bottleneck (search_news calls queue on it, agent rounds time out).
- DeepInfra serves the same model name; HTTP calls are naturally parallel,
  with no local RAM/CPU cost. Vectors must be L2-normalized client-side to
  match the local backend (sentence-transformers normalizes; the API may
  or may not).

Usage:
    from qobench.infra.embedding import embed_query
    vec = embed_query("Apple acquired Beats in 2014")
    assert len(vec) == 2560
"""
from __future__ import annotations

import os
import threading
import time
from typing import List

from qobench import config

# ---------- backend dispatch ----------

_BACKEND = os.environ.get("QOBENCH_EMBED_BACKEND", "local").lower()
if _BACKEND not in ("local", "deepinfra"):
    raise ValueError(
        f"Unknown QOBENCH_EMBED_BACKEND={_BACKEND!r}; expected 'local' or 'deepinfra'"
    )
print(f"[qobench.infra.embedding] backend={_BACKEND}", flush=True)


def embed_query(text: str) -> List[float]:
    """Encode one query string. Returns a 2560-dim float list, L2-normalized."""
    if _BACKEND == "deepinfra":
        return _embed_deepinfra(text)
    return _embed_local(text)


# Qwen3-Embedding is asymmetric and instruction-tuned: queries must be wrapped
# with this template, documents must NOT. the upstream pipeline indexed corpus chunks raw,
# matching the documented Qwen3 convention. Encoding queries raw silently
# misaligns the query vector with the document space — direct test (Phase 1.3
# on 2026-05-08) showed top-10 dropped from 0/10 to 2/10 relevant on
# t02_q001 (APA M&A_rumor) when this prefix is applied.
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a financial-news search query, "
    "retrieve relevant news articles\nQuery: "
)


# ---------- local backend ----------

# Defense-in-depth lock: serialize SentenceTransformer.encode() across threads.
# LangChain BaseRetriever.ainvoke wraps the sync _get_relevant_documents in
# asyncio.to_thread, so under ReAct's per-question retrievers many threads can
# call encode() concurrently. Even with the fp32 fix below, keep this lock —
# CPU encode is the smaller cost compared to LLM round-trip + Milvus search.
_encode_lock = threading.Lock()

# Load lock + module-level cache for thread-safe lazy load. We do NOT use
# functools.lru_cache because its docs are explicit: "calls to the wrapped
# function may concurrently execute the same uncached call." Under cold-start
# concurrency (3 ReAct workers all calling embed_query simultaneously), that
# means 3 parallel model loads = 48GB fp32 weights = OOM on a typical workstation. The
# double-checked locking pattern below guarantees exactly one load.
_load_lock = threading.Lock()
_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    with _load_lock:
        if _cached_model is not None:  # double-checked locking
            return _cached_model
        _cached_model = _load_model_impl()
        return _cached_model


def _load_model_impl():
    from sentence_transformers import SentenceTransformer

    rt = config.load_runtime()
    model_name = rt["embed_model_guess"]
    if not model_name.startswith("Qwen/Qwen3-Embedding-"):
        raise RuntimeError(
            f"Unsupported embed model: {model_name!r}. "
            "Update embedding.py to handle this model."
        )
    # Force CPU on macOS: Apple MPS has a known matmul shape bug with
    # Qwen3-Embedding (LLVM ERROR: Failed to infer result type). CPU is
    # slower (~few seconds/query) but stable. CUDA users can override via
    # env var if needed.
    import torch
    device = os.environ.get("QOBENCH_EMBED_DEVICE", "cpu")
    # Force fp32 weights. Default is bf16 (HF safetensors), which under
    # concurrent encode triggers `expected m1 and m2 to have the same dtype,
    # got: float != BFloat16` mid-attention — fp32 buffers (RoPE inv_freq,
    # layernorm) get mixed with bf16 weights inconsistently across threads.
    # fp32 doubles memory (~16GB) but on CPU is actually faster than bf16
    # (CPUs lack native bf16 SIMD; bf16 → fp32 upcast on every matmul).
    print(f"Loading {model_name} on device={device} (fp32, ~16GB ram) ...")
    return SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": torch.float32},
    )


def _embed_local(text: str) -> List[float]:
    """Local Qwen3-Embedding-4B encode (fp32 on CPU, lock-serialized)."""
    model = _load_model()
    with _encode_lock:
        vec = model.encode([QWEN3_QUERY_INSTRUCTION + text], normalize_embeddings=True)[0]
    return vec.tolist()


# ---------- deepinfra backend ----------

DEEPINFRA_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
DEEPINFRA_EMBED_URL = "https://api.deepinfra.com/v1/openai/embeddings"
_DEEPINFRA_TIMEOUT_S = 90
_DEEPINFRA_RETRIES = 5
_DEEPINFRA_BACKOFF_S = (2, 4, 8, 16)  # one entry per retry boundary; len == retries - 1


def _embed_deepinfra(text: str) -> List[float]:
    """DeepInfra /v1/openai/embeddings — OpenAI-compatible. Force L2 normalize
    on the client side so output matches the local backend regardless of
    whatever the upstream serving precision returns."""
    import math
    import sys
    import requests  # match reranker.py pattern (sync HTTP)

    api_key = config.load_deepinfra_key()
    payload = {
        "model": DEEPINFRA_EMBED_MODEL,
        "input": QWEN3_QUERY_INSTRUCTION + text,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    for attempt in range(_DEEPINFRA_RETRIES):
        try:
            resp = requests.post(DEEPINFRA_EMBED_URL, json=payload, headers=headers, timeout=_DEEPINFRA_TIMEOUT_S)
            resp.raise_for_status()
            vec = resp.json()["data"][0]["embedding"]
            # L2 normalize — local backend uses normalize_embeddings=True so we
            # must match that for vector-space consistency with the Milvus index.
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0.0:
                raise RuntimeError("DeepInfra returned a zero vector")
            if attempt > 0:
                print(f"[embedding.deepinfra] succeeded after attempt {attempt+1}/{_DEEPINFRA_RETRIES}", file=sys.stderr, flush=True)
            return [v / norm for v in vec]
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < _DEEPINFRA_RETRIES - 1:
                wait = _DEEPINFRA_BACKOFF_S[attempt]
                print(f"[embedding.deepinfra] attempt {attempt+1}/{_DEEPINFRA_RETRIES} failed ({type(e).__name__}); sleeping {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
            else:
                print(f"[embedding.deepinfra] attempt {attempt+1}/{_DEEPINFRA_RETRIES} failed ({type(e).__name__}); giving up", file=sys.stderr, flush=True)
    assert last_err is not None
    raise last_err
