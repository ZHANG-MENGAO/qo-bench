# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-NC-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Extract retriever scope (date window + optional ticker) from a v2/v3-bundle question.

Verified 2026-05-22 that v3 questions reuse the same params field names
(W_start / W_end / W1_start / W1_end / W2_start / W2_end / F / F_anchor).
v3's articles.csv is byte-identical to v2's, so the substrate-side semantics
also carry over.

Centralizes three pieces of normalization that the runners would otherwise
each have to handle inline:
  - `F_anchor` (B.1.1) as a synonym for `F` when `F` is absent.
  - W1/W2 dual-window params (B.3.2) collapsed into a union window.
  - Missing `W_start` / `W_end` → broad fallback (200-year window) to preserve
    the runner's pre-v2 behavior for any template we haven't enumerated.

Pure function, no side effects.
"""
from __future__ import annotations


def extract_retriever_scope(q: dict) -> dict:
    """Return {"w_start": str, "w_end": str, "ticker": str | None}.

    The retriever uses `current_w_start` / `current_w_end` to build a Milvus
    date-range filter, and `current_ticker` (if set) for the F3 ticker
    pre-filter.
    """
    params = q.get("params", {})

    if "W1_start" in params:
        # B.3.2 dual-window template. Union the two windows; YAGNI on a
        # dual-pass retrieval until the smoke proves union is too noisy.
        w_start = min(params["W1_start"], params["W2_start"])
        w_end   = max(params["W1_end"],   params["W2_end"])
    else:
        w_start = params.get("W_start") or "1900-01-01"
        w_end   = params.get("W_end")   or "2100-12-31"

    # F is the canonical firm-pin key. F_anchor (B.1.1) is the synonym.
    # If both ever co-occur, prefer F — explicit always wins over implicit.
    ticker = params.get("F") or params.get("F_anchor") or None

    return {"w_start": w_start, "w_end": w_end, "ticker": ticker}
