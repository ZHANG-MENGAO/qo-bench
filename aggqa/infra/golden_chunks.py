# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Golden-chunk context assembly for LC-oracle (Long-Context Oracle baseline).

Spec: docs/superpowers/specs/2026-05-18-lc-oracle-baseline-design.md §6

Four responsibilities:
  1. `collect_golden_urls(question)` — gather all `article_url`s referenced by
     any `golden_chunks*` key in any GT row, deduplicated, first-occurrence order.
  2. `collect_events_with_chunks(question)` — enumerate per-event ranked chunk
     lists (closest-to-anchor first). Backbone of the budget-aware fallback so
     the "≥1 chunk per GT event" invariant can be enforced explicitly.
  3. `build_oracle_context(question, articles_index)` — strict perfect-oracle
     path: turn URLs into chunk dicts by joining against `articles.csv`. Raises
     `BundleSkewError` on URL miss (post-2026-05-19 orphan-filter, 100% should
     be corpus-resident) and `OversizeQuestionError` on > 200K tokens.
  4. `build_oracle_context_budget_aware(question, articles_index, input_budget)`
     — fallback for the 5 cap50 questions whose strict assembly exceeds the
     200K budget (B.1.2/B.1.3/B.1.4 multi-event 2019 windows; see spec §4).
     Pass 1 selects 1 chunk per GT event (smallest `|days_to_event|`); Pass 2
     round-robins remaining budget across events. design decision (2026-05-19) — preferred
     over hard-failing 5 questions to DNF because the per-event-min-1 invariant
     still gives every GT event at least one piece of attesting evidence.

`build_oracle_context` retains the hard oversize guard; `build_oracle_context_
budget_aware` raises only if even Pass-1 floor exceeds budget (impossible across
all 768 cap50 questions per the 2026-05-19 dry-run).
"""

from __future__ import annotations

from typing import Iterable


class BundleSkewError(RuntimeError):
    """A golden_chunks URL is not in articles.csv — signals bundle/corpus skew."""


class OversizeQuestionError(RuntimeError):
    """Assembled input exceeds INPUT_TOKEN_BUDGET; hard-skip to errors.jsonl."""

    def __init__(self, qid: str, n_chunks: int, input_tokens: int, budget: int):
        super().__init__(
            f"qid={qid} input_tokens={input_tokens:,} > budget={budget:,} "
            f"(n_chunks={n_chunks})"
        )
        self.qid = qid
        self.n_chunks = n_chunks
        self.input_tokens = input_tokens
        self.budget = budget


def collect_golden_urls(question: dict) -> list[str]:
    """Collect dedup'd `article_url`s referenced by a question's GT.

    Two sources are pooled (first-occurrence order):
      1. Any `golden_chunks*` key on each GT row (A.* + B.2.* + B.4.3/4.4 use
         plain `golden_chunks`). v3 multi-event templates moved all chunks under
         (2) below; the code still handles legacy `_e1/_e2/_trigger/_announce`
         suffix keys for v2-era test fixtures, but real v3 question data does
         not emit them.
      2. Each chunk nested inside `qualifying_events[*].golden_chunks` on each
         GT row. All multi-event templates route this way: B.1.2 / B.1.3 /
         B.1.4 / B.1.5 + B.3.1/3.2/3.3 + B.4.1/4.2. Without nested traversal,
         LC-oracle loads empty context for those templates and breaks the
         capability-ceiling premise.

    Empty `gt` → empty list.
    """
    seen: list[str] = []
    seen_set: set[str] = set()

    def _add(url):
        if url and url not in seen_set:
            seen.append(url)
            seen_set.add(url)

    for gt_row in question.get("gt", []) or []:
        if not isinstance(gt_row, dict):
            continue
        for key, val in gt_row.items():
            if not key.startswith("golden_chunks"):
                continue
            if not isinstance(val, list):
                continue
            for ch in val:
                if isinstance(ch, dict):
                    _add(ch.get("article_url"))
        for ev in gt_row.get("qualifying_events") or []:
            if not isinstance(ev, dict):
                continue
            for ch in ev.get("golden_chunks") or []:
                if isinstance(ch, dict):
                    _add(ch.get("article_url"))
    return seen


def collect_events_with_chunks(question: dict) -> list[tuple[str, list[dict]]]:
    """Enumerate (event_label, ranked_chunk_list) pairs for a question.

    Each tuple represents one logical GT event — the unit design requires to
    have ≥1 attesting chunk in the assembled context. Pairs come from two
    structural sources, mirroring `collect_golden_urls`:

      - Top-level `golden_chunks*` keys on each GT row (one event = one key per
        row; B.1.2 rows carry _e1 + _e2, B.1.3 carry _trigger + _followup,
        B.1.5 carry _announce + _complete, A.*/B.2.*/B.4.3/B.4.4 carry plain
        `golden_chunks`).
      - `qualifying_events[*].golden_chunks` nested entries (one event = one
        entry; B.3.1/3.2/3.3 and B.4.1/4.2).

    Each chunk list is ranked by `|days_to_event|` ascending (closest-to-anchor
    first) so callers doing budget-aware selection get the most relevant chunk
    at index 0. Chunks missing `days_to_event` sort last.
    """
    events: list[tuple[str, list[dict]]] = []
    for row_idx, gt_row in enumerate(question.get("gt", []) or []):
        if not isinstance(gt_row, dict):
            continue
        for key, val in gt_row.items():
            if not key.startswith("golden_chunks"):
                continue
            if not isinstance(val, list) or not val:
                continue
            ranked = sorted(
                (ch for ch in val if isinstance(ch, dict) and ch.get("article_url")),
                key=lambda c: abs(c.get("days_to_event") if c.get("days_to_event") is not None else 10_000),
            )
            if ranked:
                events.append((f"r{row_idx}.{key}", ranked))
        for qe_idx, ev in enumerate(gt_row.get("qualifying_events") or []):
            if not isinstance(ev, dict):
                continue
            chs = ev.get("golden_chunks") or []
            if not isinstance(chs, list) or not chs:
                continue
            ranked = sorted(
                (ch for ch in chs if isinstance(ch, dict) and ch.get("article_url")),
                key=lambda c: abs(c.get("days_to_event") if c.get("days_to_event") is not None else 10_000),
            )
            if ranked:
                slot = ev.get("slot", "?")
                events.append((f"r{row_idx}.qe{qe_idx}.{slot}", ranked))
    return events


# ---------------------------------------------------------------------------
# Token estimation + assembly
# ---------------------------------------------------------------------------

# Per spec §4: event_definitions ~3.5K + system prompt + question + chunk wrappers.
# Calibrated against actual prompt build; ±1K tolerance is fine since we have a
# 62K output reserve as cushion.
_OVERHEAD_TOKENS = 7_000


def estimate_input_tokens(chunks: list[dict], question: dict) -> int:
    """Rough input-token estimate: sum(body chars)/4 + ~7K overhead.

    Used as the oversize guard before LLM call. vLLM `usage.prompt_tokens` is
    the authoritative count after the call; this estimate is just for routing.
    """
    body_chars = sum(len(c.get("body", "") or "") for c in chunks)
    # Question text is small (avg ~150 chars) — subsumed by overhead budget.
    return body_chars // 4 + _OVERHEAD_TOKENS


def build_oracle_context(
    question: dict,
    articles_index: dict[str, dict],
    *,
    input_budget: int | None = None,
) -> list[dict]:
    """Assemble LC-oracle context for a question from its golden chunks.

    Returns a list of `{date, ticker, title, body}` chunk dicts in
    first-occurrence URL order. Raises `BundleSkewError` if any GT URL is
    absent from the corpus (post-2026-05-19 orphan-filter, this is fail-fast).
    Raises `OversizeQuestionError` if estimated input exceeds the budget.
    """
    if input_budget is None:
        from aggqa import config
        input_budget = config.LC_ORACLE_INPUT_BUDGET

    urls = collect_golden_urls(question)
    chunks: list[dict] = []
    for url in urls:
        row = articles_index.get(url)
        if row is None:
            raise BundleSkewError(
                f"URL {url!r} in GT for qid={question.get('qid')!r} but not in articles.csv"
            )
        # Bug #2 fix (2026-05-19): piggyback article_url into the title with a
        # `<url: ...>` marker. Mirrors run.py:doc_to_chunk_dict for Naive RAG.
        # The handover format_chunk has no URL slot, so packing the URL into
        # the title is the only way to surface it to the LLM. The LLM reads
        # the marker out of the chunk header and copies the URL into the
        # answer's cited_urls field — scored by eval_v2.recall_with_prov.
        title = row.get("title", "")
        if url:
            title = f"{title} <url: {url}>"
        chunks.append({
            "date": row.get("article_date", ""),
            "ticker": row.get("ticker", ""),
            "title": title,
            "body": row.get("text", ""),
        })

    tokens = estimate_input_tokens(chunks, question)
    if tokens > input_budget:
        raise OversizeQuestionError(
            qid=question.get("qid", "<unknown>"),
            n_chunks=len(chunks),
            input_tokens=tokens,
            budget=input_budget,
        )
    return chunks


# ---------------------------------------------------------------------------
# Budget-aware fallback — per-event-min-1 + round-robin
# ---------------------------------------------------------------------------

def _row_to_assembly_chunk(url: str, row: dict) -> dict:
    """Mirror build_oracle_context's chunk shaping: pack URL into title."""
    title = row.get("title", "")
    if url:
        title = f"{title} <url: {url}>"
    return {
        "date": row.get("article_date", ""),
        "ticker": row.get("ticker", ""),
        "title": title,
        "body": row.get("text", ""),
    }


def _body_tokens(row: dict) -> int:
    """Per-chunk body-token estimate, consistent with estimate_input_tokens."""
    return len(row.get("text", "") or "") // 4


def build_oracle_context_budget_aware(
    question: dict,
    articles_index: dict[str, dict],
    *,
    input_budget: int | None = None,
) -> tuple[list[dict], dict]:
    """Budget-aware fallback for the 5 cap50 oversize questions (spec §4).

    Strategy (design decision (2026-05-19)):
      Pass 1 — Per-event-min-1 floor. For each GT event (one per top-level
        `golden_chunks*` key + one per `qualifying_events[*]` entry), pick the
        chunk with smallest `|days_to_event|`. Skip if a higher-ranked chunk
        for some other event already covered the URL (chunks that attest two
        events count once but satisfy both). Invariant: after Pass 1, every
        GT event has ≥1 attesting chunk in the selected set.
      Pass 2 — Round-robin fill. Cycle through events; each event in turn
        offers its next-ranked unselected chunk. If it fits the remaining
        budget, add it; otherwise advance the event's pointer (try a
        possibly-smaller chunk next pass). Stop when a full pass adds nothing.

    Returns `(chunks, stats)` where `stats` carries assembly bookkeeping
    (`n_events`, `n_chunks_input`, `n_chunks_kept`, `n_chunks_dropped`,
    `input_tokens_estimate`, `per_event_min_1`). The runner writes the stats
    into `predictions.jsonl` so analysis can quantify the degradation.

    Raises `BundleSkewError` if any selected URL is absent from the corpus.
    Raises `OversizeQuestionError` if Pass-1 floor exceeds budget (per the
    2026-05-19 dry-run this is impossible for cap50; kept defensively for
    future bundle expansions).
    """
    if input_budget is None:
        from aggqa import config
        input_budget = config.LC_ORACLE_INPUT_BUDGET

    events = collect_events_with_chunks(question)
    if not events:
        return [], {
            "n_events": 0, "n_chunks_input": 0, "n_chunks_kept": 0,
            "n_chunks_dropped": 0, "input_tokens_estimate": _OVERHEAD_TOKENS,
            "per_event_min_1": True,
        }

    # Universe of candidate URLs in first-occurrence order (for input-count stats).
    all_urls = collect_golden_urls(question)
    n_chunks_input = len(all_urls)

    # Lookup + skew check up front so we fail fast on any missing URL.
    for url in all_urls:
        if url not in articles_index:
            raise BundleSkewError(
                f"URL {url!r} in GT for qid={question.get('qid')!r} but not in articles.csv"
            )

    selected_urls: list[str] = []          # insertion order = first-occurrence
    selected_set: set[str] = set()

    # ---- Pass 1: per-event floor ------------------------------------------
    for event_label, ranked in events:
        if any(ch["article_url"] in selected_set for ch in ranked):
            continue
        url = ranked[0]["article_url"]
        selected_urls.append(url)
        selected_set.add(url)

    # Floor-cost check. Should never trigger for cap50; defensive only.
    floor_body_tok = sum(_body_tokens(articles_index[u]) for u in selected_urls)
    floor_tok = floor_body_tok + _OVERHEAD_TOKENS
    if floor_tok > input_budget:
        raise OversizeQuestionError(
            qid=question.get("qid", "<unknown>"),
            n_chunks=len(selected_urls),
            input_tokens=floor_tok,
            budget=input_budget,
        )

    # ---- Pass 2: round-robin fill ----------------------------------------
    # Per-event pointer into its ranked chunk list. Pass 1 already consumed
    # rank 0 (or hit an already-selected URL); start from 0 and let the
    # selected-set skip-loop advance us past anything already taken.
    event_ptr = {label: 0 for label, _ in events}
    current_tok = floor_tok

    while True:
        progress = False
        for event_label, ranked in events:
            # Advance pointer past chunks already selected (free coverage).
            idx = event_ptr[event_label]
            while idx < len(ranked) and ranked[idx]["article_url"] in selected_set:
                idx += 1
            event_ptr[event_label] = idx
            if idx >= len(ranked):
                continue

            url = ranked[idx]["article_url"]
            chunk_tok = _body_tokens(articles_index[url])
            if current_tok + chunk_tok <= input_budget:
                selected_urls.append(url)
                selected_set.add(url)
                current_tok += chunk_tok
                event_ptr[event_label] = idx + 1
                progress = True
            else:
                # Too big to fit now. Advance pointer so a possibly-smaller
                # chunk gets a chance in the next pass; don't mark progress
                # (a single skip alone shouldn't keep the outer loop alive).
                event_ptr[event_label] = idx + 1
        if not progress:
            break

    chunks = [_row_to_assembly_chunk(u, articles_index[u]) for u in selected_urls]
    final_tok = estimate_input_tokens(chunks, question)
    stats = {
        "n_events": len(events),
        "n_chunks_input": n_chunks_input,
        "n_chunks_kept": len(chunks),
        "n_chunks_dropped": n_chunks_input - len(chunks),
        "input_tokens_estimate": final_tok,
        "per_event_min_1": True,
    }
    return chunks, stats
