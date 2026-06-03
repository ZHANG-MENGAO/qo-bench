# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Load the authoritative event-type definitions (`benchmark/event_definitions.md`).

`event_definitions.md` is pasted into the system prompt of every paradigm so
all 4 paradigms (long-context, RAG / Agentic RAG, GraphRAG, IE->SQL) share the
same task ontology. Definitions are SEC-grounded (Form 8-K Items 1.01 / 1.02 /
2.01 / 5.02, Securities Act Sec. 5, NYSE LCM Sec. 202.03 / 703.02, NASDAQ Rule
5250).

We prepend the document to our existing system prompts rather than replacing
them: the upstream definitions are the shared ontology; our prompts are the
paradigm-specific behavior contract layered on top.
"""
from __future__ import annotations

from functools import lru_cache

from aggqa import config


EVENT_DEFINITIONS_PATH = (
    config.PROJECT_ROOT / "benchmark" / "event_definitions.md"
)


@lru_cache(maxsize=1)
def load_event_definitions() -> str:
    if not EVENT_DEFINITIONS_PATH.exists():
        raise RuntimeError(
            f"event_definitions.md not found at {EVENT_DEFINITIONS_PATH}; "
            "expected it under the bundle's benchmark/ directory."
        )
    return EVENT_DEFINITIONS_PATH.read_text(encoding="utf-8")


def _strip_universal_schema_section(text: str) -> str:
    """Remove the `## Schema (universal record)` section from event_definitions.md.

    The universal-record schema documented in that section applies only to
    the 15 standard templates; B.1.2/3/5/6 + B.4.3 use variant schemas with
    different field names (e1_anchor_date, gap_days, buyer_ticker, bucket,
    days_between, ...). Keeping the universal description in system prompt
    would contradict the per-template schema injected later in the user
    prompt. Output schema therefore lives in user prompt only — see
    `infra/output_schemas.py`. This function strips the section so the
    system prompt carries only the task-invariant ontology (event-type
    definitions, sources, regulatory alignment).
    """
    start_marker = "## Schema (universal record)"
    end_marker = "## Event types"
    i = text.find(start_marker)
    if i == -1:
        return text  # tolerate future schema-less event_definitions.md
    j = text.find(end_marker, i)
    if j == -1:
        return text
    return text[:i] + text[j:]


ADDITIONAL_RULES = """

---

## Additional rules (apply to every question)

1. **Deduplication.** If multiple articles describe the same underlying event
   (same firm, same event type, same anchor date), emit ONE record for that
   event — not one per supporting article. Combine the URLs from all matching
   articles into the record's `cited_urls` list. Cap50 questions have a
   median of 9 supporting chunks per event; redundant coverage of the same
   M&A deal across press release + earnings call + analyst note is the norm.

2. **Anchor date selection.** Each event type above defines what
   `anchor_date` represents (press release date for M&A_announce; closing
   date for M&A_complete; termination date for M&A_cancel; article
   publication date for M&A_rumor / CEO_change / CFO_change / IPO;
   effective date for Stock_split).

   When the article body states the canonical date explicitly
   ("Acme announced on January 29 that..."), use that date. Otherwise,
   use the article's own publication date — shown as the `YYYY-MM-DD`
   after `[N]` in each chunk header. For CEO_change / CFO_change / IPO /
   M&A_rumor, the publication date IS the canonical anchor by convention
   above. For the other types, the publication date is an acceptable
   fallback (typically within 1–2 days of the canonical date).

   Do NOT guess or invent dates from prior knowledge of corporate events.
   Use only dates that are explicitly stated in the article body or shown
   in the chunk header.
"""


def with_event_definitions(base_system_prompt: str) -> str:
    """Prepend the upstream event definitions to a base system prompt.

    Used by run.py (Naive RAG), react_agent.py (ReAct), and lc_oracle_run.py
    (LC-oracle) to inject the v2 task ontology. The `## Schema (universal
    record)` section is stripped (moved to user prompt per-template; see
    docstring of `_strip_universal_schema_section`). Separator is a `---`
    horizontal rule so the model sees a clear boundary between shared
    ontology and paradigm behavior contract.

    `ADDITIONAL_RULES` (dedup + date-hallucination guard) is appended at
    the end so the rules sit close to where the LLM starts generating,
    maximizing attention on them. Both rules are task-invariant across all
    20 templates and all 3 paradigms.
    """
    definitions = _strip_universal_schema_section(load_event_definitions())
    return definitions + "\n\n---\n\n" + base_system_prompt + ADDITIONAL_RULES
