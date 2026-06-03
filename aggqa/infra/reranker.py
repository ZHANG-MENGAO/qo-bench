# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""DeepInfra reranker REST client (Qwen3-Reranker-4B).

Errors PROPAGATE — callers must not silently fall back to unranked retrieval.
DeepInfra outages should surface in run_log.txt and produce empty predictions
for the affected questions, not pollute scored numbers with unranked context.

Endpoint contract (verified 2026-05-13 via probe):
  POST https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B
  Headers: Authorization: Bearer <key>, Content-Type: application/json
  Body:    {"queries": [str], "documents": list[str]}
           NOTE: `queries` is plural and must be a single-element list; the
           server returns 422 for the singular `query` or for documents
           nested per-query. Scores are parallel to documents.
  Return:  {"scores": list[float], "inference_status": {...}, ...}
"""
from __future__ import annotations

import asyncio
import time

import requests

from aggqa import config


# Transient network errors worth retrying. Application errors (HTTPError
# from raise_for_status() — i.e., 4xx/5xx) are NOT retried because those
# signal rate limits, malformed requests, or server bugs that masking with
# retry would just hide.
_TRANSIENT_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def rerank_sync(query: str, documents: list[str], api_key: str) -> list[float]:
    """Synchronous rerank. Returns scores parallel to `documents`.

    Retries up to config.RERANK_RETRY_MAX_ATTEMPTS times on transient
    network errors (connection reset, timeout, chunked-encoding glitch).
    Raises requests.HTTPError on any non-2xx response (no retry). Caller
    sorts and truncates.
    """
    if not documents:
        return []
    url = f"{config.DEEPINFRA_BASE_URL}/{config.RERANK_MODEL}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"queries": [query], "documents": documents}

    last_exc: Exception | None = None
    for attempt in range(1, config.RERANK_RETRY_MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=config.RERANK_TIMEOUT_S
            )
            resp.raise_for_status()
            return list(resp.json()["scores"])
        except _TRANSIENT_EXCEPTIONS as e:
            last_exc = e
            if attempt >= config.RERANK_RETRY_MAX_ATTEMPTS:
                raise
            # Linear backoff: 1.5s, 3.0s by default.
            time.sleep(config.RERANK_RETRY_BACKOFF_S * attempt)
    raise last_exc  # unreachable


async def rerank(query: str, documents: list[str], api_key: str) -> list[float]:
    """Async wrapper — runs rerank_sync in a worker thread."""
    return await asyncio.to_thread(rerank_sync, query, documents, api_key)
