# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-NC-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Per-template output schema strings injected into the user prompt.

Source of truth: templates_config.json[<id>].output_schema (paper A/B IDs).

The schema for an LLM-output record varies by template. Three families:
  - STANDARD (event-tuple)            : A.1.1, A.2.1, B.1.1, B.2.1, B.2.2,
                                        B.4.4, and the A.3.1 event-tuple form
  - ENTITY_PROJECTION (firm-only)     : A.1.2, B.3.1/3.2/3.3, B.4.1, B.4.2,
                                        and the A.3.1 entity-projection form
  - Variant (per-template custom)     : B.1.2, B.1.3, B.1.4, B.1.5, B.4.3

The schema is part of the *user* prompt rather than the system prompt:

1. It's per-question (depends on template_id), not task-invariant.
2. Long-context attention: placed close to the question to minimize drift in
   LC-oracle (system at tok 0, question at tok 100K+, schema-near-question
   keeps the contract in working memory).
3. The system prompt's universal-record description was stripped (see
   `event_definitions.with_event_definitions`) to avoid contradicting the
   variant schemas for B.1.2 / B.1.3 / B.1.4 / B.1.5 / B.4.3.

ENTITY_PROJECTION_SCHEMA was added 2026-05-21 (this file's commit) because
STANDARD_SCHEMA contradicted the rendered question for the 7 entity-projection
templates ("Return distinct firm tickers" vs. STANDARD's 6-field event-tuple).
The LLM resolved the contradiction 50/50, and runs that picked List[str]
shape were silently zero'd by parse_model_output.

Maintenance: when the benchmark bundle changes, re-read the GT row format
and templates_config.json[<id>].output_schema and update below.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema strings
# ---------------------------------------------------------------------------

STANDARD_SCHEMA = """# Output schema

Return a JSON array of event records. Each record has fields:
  firm_ticker: string (uppercase ticker)
  event_type: one of [M&A_announce, M&A_complete, M&A_cancel, M&A_rumor,
              CEO_change, CFO_change, IPO, Stock_split]
  anchor_date: "YYYY-MM-DD"
  role: "buyer" | "seller" | "target" | null
        For M&A_announce / M&A_complete / M&A_cancel: emit one record from
        each public-firm participant's perspective with role set accordingly:
          - "buyer" when the firm is the acquirer (use this when the firm
            acquired another, even if the target is also publicly traded —
            buyer-side is the dominant GT convention).
          - "seller" or "target" when the firm is being acquired.
        For M&A_rumor: role = null (GT convention; do not emit "target").
        For single-participant events (CEO_change, CFO_change, IPO,
        Stock_split): role = null.
  counterparty_ticker: string | null
        For M&A multi-stage events: the opposite-role participant's ticker
        if mentioned in the articles, otherwise null. Most GT records leave
        this null; do not invent a ticker if it isn't in the articles.
  cited_urls: array of strings
        URLs of articles that support this event. Copy from the `<url: ...>`
        marker in the corresponding chunk's header line. May be empty if no
        supporting article URL is present in the prompt, but populate it
        whenever the URL is available (scored by eval_v2.recall_with_prov).

If no events match the question, return [].
"""

B12_SCHEMA = """# Output schema

Return a JSON array of event-pair records. Each record has fields:
  firm_ticker: string
  e1_anchor_date: "YYYY-MM-DD"  (the earlier event's anchor date)
  e2_anchor_date: "YYYY-MM-DD"  (the later event's anchor date)
  gap_days: integer             (e2_anchor_date minus e1_anchor_date, in days)
  cited_urls: array of strings  (URLs of articles that support BOTH events;
                                 copy from the `<url: ...>` marker in each
                                 chunk's header line. May be empty if no
                                 supporting URLs are present in the prompt.)

If no firm has the requested pair of events in the window, return [].
"""

B13_SCHEMA = """# Output schema

Return a JSON array of trigger-followup records. Each record has fields:
  firm_ticker: string
  e_trigger_anchor_date: "YYYY-MM-DD"   (the triggering event)
  e_followup_anchor_date: "YYYY-MM-DD"  (the follow-up event)
  lag_days: integer                     (e_followup minus e_trigger, in days)
  cited_urls: array of strings          (URLs of articles that support BOTH
                                         events; copy from the `<url: ...>`
                                         marker in each chunk's header line.
                                         May be empty if no supporting URLs
                                         are present in the prompt.)

If no firm has the requested trigger/followup pattern in the window, return [].
"""

B15_SCHEMA = """# Output schema

Return a JSON array of deal records. Each record has fields:
  buyer_ticker: string                 (the acquirer's stock ticker,
                                        uppercase, e.g. "EXAS")
  target_ticker: string                (the target firm's FULL COMPANY
                                        NAME as it appears in the article,
                                        e.g. "Genomic Health, Inc." — NOT
                                        a ticker symbol, even when the
                                        target is publicly traded. This
                                        field is named "target_ticker"
                                        for legacy reasons but its
                                        semantic is the company name.)
  announce_date: "YYYY-MM-DD"          (M&A_announce anchor date)
  close_date: "YYYY-MM-DD"             (M&A_complete anchor date)
  cited_urls: array of strings         (URLs of articles that support the
                                        deal; copy from the `<url: ...>`
                                        marker in each chunk's header line.
                                        May be empty if no supporting URLs
                                        are present in the prompt.)

If no deals match, return [].
"""

B16_SCHEMA = """# Output schema

Return a JSON array containing exactly ONE object (or [] if you cannot
determine both events). The object has fields:
  firm_ticker: string (uppercase ticker)
  e1_anchor_date: "YYYY-MM-DD"  (anchor of the FIRST event type named in
                                 the question — the one after "its")
  e2_anchor_date: "YYYY-MM-DD"  (anchor of the SECOND event type named
                                 in the question)
  days_between: integer         (e2_anchor_date minus e1_anchor_date;
                                 SIGNED — may be negative if event 2
                                 occurred before event 1)

If you cannot determine both events from the articles, return [].
"""

ENTITY_PROJECTION_SCHEMA = """# Output schema

Return a JSON array of firm records, one per distinct firm. Each record has
fields:
  firm_ticker: string (uppercase ticker)
  cited_urls: array of strings
        URLs of articles that support this firm matching the question.
        Copy from the `<url: ...>` marker in each supporting chunk's header
        line. May be empty if no supporting URLs are present in the prompt;
        populate when available (scored by eval_v2.recall_with_prov).

If no firms match the question, return [].
"""


B43_SCHEMA = """# Output schema

Return a JSON array of event records, each tagged with a time bucket.
Each record has fields:
  firm_ticker: string
  event_type: one of [M&A_announce, M&A_complete, M&A_cancel, M&A_rumor,
              CEO_change, CFO_change, IPO, Stock_split]
  anchor_date: "YYYY-MM-DD"
  role: "buyer" | "seller" | "target" | null
        For M&A_announce / M&A_complete / M&A_cancel: emit "buyer" when the
        firm is the acquirer; "seller" or "target" when acquired. Buyer is
        the dominant GT convention. For M&A_rumor: null. For single-
        participant events (CEO_change / CFO_change / IPO / Stock_split): null.
  counterparty_ticker: string | null
        Opposite-role ticker if in articles, else null.
  cited_urls: array of strings
        Copy from `<url: ...>` markers in chunk headers.
  bucket: string                       (Time bucket label derived from
                                        anchor_date. The granularity is
                                        specified in the question text
                                        ("grouped by week/month/quarter").
                                        Required formats:
                                          - week:    ISO week, e.g. "2020-W23"
                                            (4-digit year, "-W", 2-digit ISO
                                            week 01-53, zero-padded).
                                          - month:   "YYYY-MM" e.g. "2020-06".
                                          - quarter: "YYYY-Qn" e.g. "2020-Q2".
                                        Pick the format that matches the
                                        bucket_unit asked for in the question.)

If no events match, return [].
"""


# ---------------------------------------------------------------------------
# Template → schema mapping
# ---------------------------------------------------------------------------

# Templates that use the standard event-tuple schema. The A.3.1 event-tuple
# phrasing is also routed here via get_schema_for_template's legacy_id branch.
STANDARD_TEMPLATES: tuple[str, ...] = (
    "A.1.1", "A.2.1",
    "B.1.1",
    "B.2.1", "B.2.2",
    "B.4.4",
)

# Templates whose v3 templates_config.json output_schema is
# {firm_ticker (required) + cited_urls (optional)} only. The rendered question
# also asks "Return distinct firm tickers", so we give the LLM a minimal schema
# matching that intent. A.3.1 T18 form is routed here via the legacy_id branch.
ENTITY_PROJECTION_TEMPLATES: tuple[str, ...] = (
    "A.1.2",
    "B.3.1", "B.3.2", "B.3.3",
    "B.4.1", "B.4.2",
)

# Templates that use a variant schema (per schema.md §"Variants for
# multi-event GT rows").
CUSTOM_TEMPLATES: dict[str, str] = {
    "B.1.2": B12_SCHEMA,
    "B.1.3": B13_SCHEMA,
    "B.1.4": B15_SCHEMA,   # deal-pairing (announce->complete)
    "B.1.5": B16_SCHEMA,   # days-between composite (composite_b16)
    "B.4.3": B43_SCHEMA,
}


def get_schema_for_template(template_id: str, legacy_id: str | None = None) -> str:
    """Return the user-prompt output-schema block for one template.

    A.3.1 carries two phrasing forms (templates_config.json §A.3.1.description):
      - T18: Entity-projection ("Which firms played role R..."). Returns just
        firm tickers; uses ENTITY_PROJECTION_SCHEMA.
      - T40: Event-projection ("List events of type E where role R was
        reported..."). Returns full event tuples; uses STANDARD_SCHEMA.
    Other templates ignore `legacy_id`.

    Raises ValueError for unknown templates so callers fail-fast on a new
    bundle that adds templates we haven't documented.
    """
    if template_id == "A.3.1":
        if legacy_id == "T40":
            return STANDARD_SCHEMA
        # Default T18 (Entity-projection). Matches question_renderer's default.
        return ENTITY_PROJECTION_SCHEMA
    if template_id in CUSTOM_TEMPLATES:
        return CUSTOM_TEMPLATES[template_id]
    if template_id in ENTITY_PROJECTION_TEMPLATES:
        return ENTITY_PROJECTION_SCHEMA
    if template_id in STANDARD_TEMPLATES:
        return STANDARD_SCHEMA
    raise ValueError(
        f"Unknown template_id: {template_id!r}. Update output_schemas.py "
        "to match benchmark/templates_config.json output_schema entries."
    )
