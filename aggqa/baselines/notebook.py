# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Per-question notebook tools and finalization for the ReAct agent.

This module owns:
  - load_template_cfg(template_id): pull one template's config dict
  - finalize_notebook(notebook, template_cfg): deterministic Python answer
    assembly — dedup by identity_keys, dispatch by scoring type.
  - build_candidate_schema(), make_record_candidates_tool(),
    make_record_b16_answer_tool().

The notebook is a per-question list[dict] (closure-captured by tools). Tools
append dicts; finalize_notebook reads & collapses.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from langchain_core.tools import StructuredTool, ToolException, tool as lc_tool
from pydantic import BaseModel, ConfigDict, Field, create_model

from aggqa import config


# v3 retired A.3.combined entirely (questions are now keyed A.3.1; T18 and
# T40 both fold into A.3.1 with looser identity_keys=['firm_ticker']).
# Alias kept as defensive no-op: if an LLM emits a legacy "A.3.combined"
# id (rare; only happens if an example in the prompt or fine-tuning data
# carried it), we still route to the right config rather than KeyError.
TEMPLATE_ALIASES = {
    "A.3.combined": "A.3.1",
}


@lru_cache(maxsize=1)
def _load_all_templates() -> dict:
    """Read templates_config.json once per process. Excludes the '_meta' key."""
    raw = json.loads(config.TEMPLATES_CONFIG_PATH.read_text())
    return {tid: tcfg for tid, tcfg in raw.items() if not tid.startswith("_")}


def load_template_cfg(template_id: str) -> dict:
    """Return the config dict for one template_id, after applying alias map.
    Raises KeyError if neither the id nor its alias is known."""
    resolved = TEMPLATE_ALIASES.get(template_id, template_id)
    all_cfg = _load_all_templates()
    if resolved not in all_cfg:
        raise KeyError(f"Unknown template_id: {template_id!r}")
    return all_cfg[resolved]


def finalize_notebook(notebook: list[dict], template_cfg: dict):
    """Convert per-question notebook to predictions.jsonl-shaped answer.

    Dispatch by template_cfg['scoring']:
      - empty_check  -> returns the raw notebook contents (no dedup).
                        Empty notebook → []; any hallucinated record →
                        non-empty answer → eval gives 0. This is the
                        signal we want to measure (hallucination
                        resistance), not silence it.
      - int_match    -> returns int of the LAST entry's 'value' field;
                        if the notebook holds list-shaped candidates
                        instead (wrong tool used), returns 0.
                        Empty notebook → 0.
      - list_recall  -> dedup by identity_keys (+ sorted_keys when set);
                        union cited_urls within a key; first-wins for
                        other fields. Entries that look like int-tool
                        records (have 'value', no firm_ticker) are
                        skipped so wrong-tool calls don't pollute.
    """
    scoring = template_cfg.get("scoring")

    if scoring == "empty_check":
        # Strip int-tool shaped entries (defensive — they'd be wrong-tool
        # calls; for empty_check we don't want to count "value": 0 as a
        # hallucinated event row).
        return [
            dict(item) for item in notebook
            if "firm_ticker" in item or "event_type" in item
        ]

    if scoring == "int_match":
        if not notebook:
            return 0
        # Walk in reverse for the most-recent int-shaped entry.
        for item in reversed(notebook):
            if "value" in item:
                return int(item.get("value", 0))
        return 0

    if scoring == "composite_b16":
        # B.1.5: 4-field dict wrapped in single-element list per output_schema.
        # Walk in reverse for the most-recent b16-shaped entry (later tool calls
        # overwrite earlier ones — same semantics as int_match).
        empty_dict = {
            "firm_ticker": None,
            "e1_anchor_date": None,
            "e2_anchor_date": None,
            "days_between": 0,
        }
        if not notebook:
            return [empty_dict]
        for item in reversed(notebook):
            if "days_between" in item and "e1_anchor_date" in item:
                # Coerce string days_between to int (LLMs sometimes emit "100").
                db = item.get("days_between", 0)
                try:
                    db_int = int(db) if db is not None else 0
                except (ValueError, TypeError):
                    db_int = 0
                return [{
                    "firm_ticker": item.get("firm_ticker"),
                    "e1_anchor_date": item.get("e1_anchor_date"),
                    "e2_anchor_date": item.get("e2_anchor_date"),
                    "days_between": db_int,
                }]
        return [empty_dict]

    if scoring != "list_recall":
        raise ValueError(f"Unknown scoring type: {scoring!r}")

    # Drop int-tool records before list_recall dedup (wrong tool used).
    notebook = [item for item in notebook if "value" not in item or "firm_ticker" in item]

    identity_keys = template_cfg.get("identity_keys") or []
    sorted_keys = template_cfg.get("identity_keys_sorted") or []
    # identity_keys_sorted: currently only B.1.2 sets this
    # (verified 2026-05-14: ['e1_anchor_date', 'e2_anchor_date']).

    seen: dict[tuple, dict] = {}
    for item in notebook:
        key_parts = tuple(item.get(k) for k in identity_keys)
        if sorted_keys:
            sk_values = [item.get(k) for k in sorted_keys]
            sk_tuple = tuple(
                sorted(sk_values, key=lambda x: (x is None, x))
            )
            key_parts = key_parts + (sk_tuple,)

        if key_parts in seen:
            existing_urls = set(seen[key_parts].get("cited_urls") or [])
            new_urls = set(item.get("cited_urls") or [])
            seen[key_parts]["cited_urls"] = sorted(existing_urls | new_urls)
        else:
            seen[key_parts] = dict(item)
            # Normalize cited_urls to a sorted list even on first insert,
            # so output is deterministic regardless of insertion order.
            urls = seen[key_parts].get("cited_urls") or []
            seen[key_parts]["cited_urls"] = sorted(set(urls))

    return list(seen.values())


# Map JSON-Schema scalar types -> Python types. Arrays of strings → list[str].
# Anything more exotic is out of scope (the templates only use str/int/array<str>/nullable).
_JSON_TYPE_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _resolve_field_type(spec: dict) -> type:
    """Translate one JSON-Schema property spec to a Python type.

    Handles:
      - "type": "string"  -> str
      - "type": ["string", "null"]  -> Optional[str]
      - "type": "array", "items": {"type": "string"}  -> list[str]
      - enums (we keep the underlying scalar type and let pydantic enforce values via
        Literal if needed in a future iteration; current contract uses str)
    """
    t = spec.get("type")
    if isinstance(t, list):
        # Nullable union, e.g. ["string", "null"]
        non_null = [x for x in t if x != "null"]
        if len(non_null) == 1 and non_null[0] in _JSON_TYPE_TO_PY:
            return Optional[_JSON_TYPE_TO_PY[non_null[0]]]
        return Optional[str]   # conservative fallback
    if t == "array":
        item_t = spec.get("items", {}).get("type", "string")
        return list[_JSON_TYPE_TO_PY.get(item_t, str)]
    return _JSON_TYPE_TO_PY.get(t, str)


_DEFAULT_EVENT_SCHEMA_PROPERTIES: dict = {
    "firm_ticker": {"type": "string"},
    "event_type": {"type": "string"},
    "anchor_date": {"type": "string", "format": "date"},
    "role": {"type": ["string", "null"]},
    "counterparty_ticker": {"type": ["string", "null"]},
    "cited_urls": {"type": "array", "items": {"type": "string"}},
}
_DEFAULT_EVENT_SCHEMA_REQUIRED: set[str] = {"firm_ticker", "event_type", "anchor_date"}


def build_candidate_schema(template_cfg: dict) -> type[BaseModel]:
    """Build a pydantic BaseModel for one template's record_candidates payload.

    Field set is derived from template_cfg['output_schema']['items']:
      - 'required' list -> required fields (no default)
      - other 'properties' -> Optional fields with default None
      - 'cited_urls' is always required AND must be a non-empty list[str]
        (overrides the JSON Schema, which allows empty arrays).

    Fallback when a template's output_schema has no items.properties (e.g.
    maxItems=0): under uniform tool binding we still need a usable
    record_candidates schema so the model has a real surface to answer onto.
    We fall back to a generic event schema (the union of fields the A.x.x
    list_recall templates use).

    Pydantic config: extra='forbid' so a field-name drift (e.g. `firm` instead of
    `firm_ticker`) raises ValidationError immediately — the bug commit d341283
    addressed at the prompt layer, now enforced structurally.
    """
    items = template_cfg.get("output_schema", {}).get("items", {})
    properties: dict = items.get("properties", {})
    required: set[str] = set(items.get("required", []))
    if not properties:
        # empty_check fallback — see docstring.
        properties = dict(_DEFAULT_EVENT_SCHEMA_PROPERTIES)
        required = set(_DEFAULT_EVENT_SCHEMA_REQUIRED)
    # cited_urls is always required (per spec §3.2 provenance contract).
    required.add("cited_urls")

    fields: dict[str, tuple] = {}
    for name, spec in properties.items():
        if name == "cited_urls":
            fields[name] = (list[str], Field(..., min_length=1))
            continue
        py_t = _resolve_field_type(spec)
        if name in required:
            fields[name] = (py_t, Field(...))
        else:
            # Optional with default None
            fields[name] = (Optional[py_t], Field(default=None))

    # If the schema declares cited_urls only in properties (it does for every
    # list_recall template), we covered it above. If it doesn't (defensive),
    # add it.
    if "cited_urls" not in fields:
        fields["cited_urls"] = (list[str], Field(..., min_length=1))

    model_name = f"Candidate_{template_cfg.get('legacy_id', 'X')}"
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _format_notebook_snapshot(notebook: list[dict]) -> str:
    """Compact textual rendering of the current notebook for the model.

    Each entry is one line: '<idx>. <key=val ...>'. The model reads this back
    on its next turn so it can avoid duplicates and judge whether to keep going.
    """
    if not notebook:
        return "Notebook is empty."
    lines = []
    for i, item in enumerate(notebook, start=1):
        # Stable rendering: keys sorted, cited_urls collapsed to count.
        pairs = []
        for k in sorted(item.keys()):
            if k == "cited_urls":
                pairs.append(f"cited_urls=[{len(item[k])} url(s)]")
            else:
                pairs.append(f"{k}={item[k]!r}")
        lines.append(f"{i}. " + " | ".join(pairs))
    return "Current notebook:\n" + "\n".join(lines)


def make_record_candidates_tool(notebook: list[dict],
                                 seen_urls: set[str],
                                 schema_cls: type[BaseModel],
                                 debug_dump_path=None):
    """StructuredTool that advertises per-template schema to the LLM.

    The args_schema is built dynamically from schema_cls so the OpenAI tool-call
    API receives properties / required / additionalProperties=false for THIS
    template. Model sees the constraint at generation time and conforms (rather
    than us rejecting drift post-hoc with a ToolException, which crashed Q1 in
    the 2026-05-14 smoke).

    Closure-captured state (per-question):
      - notebook: list[dict] — appended to on successful record
      - seen_urls: set[str] — populated by search_news; used for cited_url check
      - schema_cls: pydantic class built from the template's output_schema
      - debug_dump_path: if set, every appended item is also written as a JSONL
        line for redundancy-pattern diagnostics (bug #2 investigation). Off by
        default.

    Tool semantics (unchanged):
      - All items in one call are validated FIRST. Atomic on failure.
      - Return value is the full notebook snapshot.

    Runtime validation kept as defense in depth: even though args_schema is
    advertised at the API layer, OpenRouter passthrough to non-OpenAI providers
    (qwen, etc.) may not enforce additionalProperties=false strictly. So we
    still run schema_cls(**raw) on each item, plus the provenance check.
    """
    call_counter = {"n": 0}

    # Wrap the per-template Candidate class inside a single-field args wrapper
    # so the tool's top-level signature is `items: list[Candidate_TXX]`. The
    # __name__ is made unique to avoid pydantic class-cache collisions across
    # concurrent questions.
    args_cls = create_model(
        f"RecordCandidatesArgs_{schema_cls.__name__}",
        __config__=ConfigDict(extra="forbid"),
        items=(list[schema_cls], Field(
            ..., min_length=1,
            description=(
                "List of candidate events identified from the most recent "
                "search_news round. Each item must conform to this template's "
                "schema and must cite at least one URL from observed chunks."
            ),
        )),
    )

    async def _impl(items) -> str:
        """Record candidates extracted from the most recent search_news round.
        Each item must conform to the per-template schema and must cite at
        least one URL from the article chunks observed so far. Returns the
        full current notebook contents."""
        # LangChain's StructuredTool with args_schema MAY auto-instantiate
        # nested pydantic types when serializing args to the function, OR may
        # pass dicts. Handle both. Runtime validation is mandatory anyway as
        # the defensive net (see docstring).
        validated = []
        for raw in items:
            if isinstance(raw, schema_cls):
                inst = raw
            elif isinstance(raw, dict):
                try:
                    inst = schema_cls(**raw)
                except Exception as e:
                    raise ToolException(
                        f"Schema validation failed for item {raw!r}: {e}"
                    ) from e
            else:
                raise ToolException(
                    f"Unexpected item type: {type(raw).__name__}. "
                    f"Expected dict or {schema_cls.__name__}."
                )
            for url in inst.cited_urls:
                if url not in seen_urls:
                    raise ToolException(
                        f"cited_url not in observed chunks: {url}. "
                        f"You must only cite URLs returned by search_news."
                    )
            validated.append(inst)

        call_counter["n"] += 1
        for inst in validated:
            item_dump = inst.model_dump()
            notebook.append(item_dump)
            if debug_dump_path is not None:
                with open(debug_dump_path, "a") as f:
                    f.write(json.dumps({
                        "record_call_n": call_counter["n"],
                        "item": item_dump,
                    }) + "\n")
        return _format_notebook_snapshot(notebook)

    return StructuredTool.from_function(
        coroutine=_impl,
        name="record_candidates",
        description=(
            "Record candidates extracted from the most recent search_news round. "
            "Each item must conform to the per-template schema and must cite at "
            "least one URL from the article chunks observed so far. Returns the "
            "full current notebook contents."
        ),
        args_schema=args_cls,
    )


def make_record_b16_answer_tool(notebook: list[dict], seen_urls: set[str]):
    """Async @tool for B.1.5 (composite_b16 templates).

    Records a 4-field composite answer (firm_ticker, e1_anchor_date,
    e2_anchor_date, days_between) plus optional citations. Multiple calls
    overwrite — finalize_notebook walks in reverse and takes the most-recent
    b16-shaped entry. Provenance check fires only for cited URLs actually
    supplied (cited_urls may be empty since B.1.5 has supports_provenance=False).
    """
    @lc_tool
    async def record_b16_answer(
        firm_ticker: str,
        e1_anchor_date: str,
        e2_anchor_date: str,
        days_between: int,
        cited_urls: list[str],
    ) -> str:
        """Record the composite B.1.5 answer for this question.

        - firm_ticker: the firm under analysis (matches question params.F)
        - e1_anchor_date: anchor date of the first event (E1), ISO YYYY-MM-DD
        - e2_anchor_date: anchor date of the second event (E2), ISO YYYY-MM-DD
        - days_between: signed integer, anchor_date(E2) − anchor_date(E1) in days
        - cited_urls: URLs from search_news chunks that ground both anchor dates;
          may be empty (B.1.5 has supports_provenance=False).
        """
        for url in cited_urls:
            if url not in seen_urls:
                raise ToolException(
                    f"cited_url not in observed chunks: {url}. "
                    f"You must only cite URLs returned by search_news."
                )
        notebook.append({
            "firm_ticker": firm_ticker,
            "e1_anchor_date": e1_anchor_date,
            "e2_anchor_date": e2_anchor_date,
            "days_between": int(days_between),
            "cited_urls": list(cited_urls),
        })
        return (
            f"Recorded b16_answer: {firm_ticker} "
            f"e1={e1_anchor_date} e2={e2_anchor_date} days={int(days_between)}. "
            f"Notebook size: {len(notebook)}."
        )

    return record_b16_answer
