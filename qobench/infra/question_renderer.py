# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Render an eval-bundle question record into the LLM-visible question text.

v2 bundles (2026-05-12 .. 2026-05-19) shipped a top-level `nl_question` field
with the upstream canonical phrasing baked in. v3 (2026-05-21) dropped that field
from the question JSONL — we render from `id` + `params` ourselves.

This module is designed to reproduce the upstream v2 `nl_question` strings exactly
when given v2 params, so scoring on v2 vs v3 can be apples-to-apples. Verified
byte-identical across all 762 v2 cap50 questions (see tests/test_question_renderer.py).

A.3.1 has two phrasing forms determined by `legacy_id`:
  - T18 (Entity projection): "Which firms played role ... Return distinct firm tickers."
  - T40 (Event projection):  "List events of type ... where role ... was reported"
"""
from __future__ import annotations


def render_question(q: dict) -> str:
    """Return the LLM-visible question text for one bundle record.

    Backward compatible: if the record already carries `nl_question` (v2 bundle),
    return it verbatim. Otherwise render from `id` + `params` (v3 bundle).

    Raises KeyError if the template id is unknown OR a required param is missing.
    """
    if "nl_question" in q:
        return q["nl_question"]

    template_id = q.get("id")
    p = q.get("params", {}) or {}
    legacy_id = q.get("legacy_id")

    if template_id == "A.1.1":
        return (
            f"List all events of type {p['E']} between "
            f"{p['W_start']} and {p['W_end']}."
        )
    if template_id == "A.1.2":
        return (
            f"Which firms had at least one event of type {p['E']} between "
            f"{p['W_start']} and {p['W_end']}? Return distinct firm tickers."
        )
    if template_id == "A.2.1":
        return (
            f"List events of type {p['E']} where firm {p['F']} is a participant, "
            f"between {p['W_start']} and {p['W_end']}."
        )
    if template_id == "A.3.1":
        if legacy_id == "T40":
            return (
                f"List events of type {p['E']} where role {p['R']} was reported "
                f"on a participant, between {p['W_start']} and {p['W_end']}."
            )
        # Default to T18 (Entity-projection) form
        return (
            f"Which firms played role {p['R']} in events of type {p['E']} "
            f"between {p['W_start']} and {p['W_end']}? Return distinct firm tickers."
        )
    if template_id == "B.1.1":
        F = p["F_anchor"]
        E = p["E"]
        return (
            f"List events of type {E} that occurred {p['direction']} "
            f"{F}'s anchor event of type {E}, where the anchor and the candidate "
            f"events all fall between {p['W_start']} and {p['W_end']}. Exclude {F} itself."
        )
    if template_id == "B.1.2":
        return (
            f"Between {p['W_start']} and {p['W_end']}, list firms where an event "
            f"of type {p['E1']} and an event of type {p['E2']} occurred within "
            f"{p['gap_days']} days of each other (any order). One row per firm "
            f"with both anchor dates and the gap in days."
        )
    if template_id == "B.1.3":
        return (
            f"Between {p['W_start']} and {p['W_end']}, list firms where an event "
            f"of type {p['E_trigger']} was followed by an event of type "
            f"{p['E_followup']} within {p['lag_days']} days. One row per firm "
            f"with both anchor dates and the lag in days."
        )
    if template_id == "B.1.4":
        return (
            f"List M&A deals announced between {p['W_start']} and {p['W_end']} "
            f"that completed within {p['horizon_days']} days of the announcement. "
            f"One row per deal with buyer ticker, target ticker, announce date, "
            f"close date."
        )
    if template_id == "B.1.5":
        return (
            f"For firm {p['F']}, how many days elapsed between its {p['E1']} "
            f"event and its {p['E2']} event, between {p['W_start']} and "
            f"{p['W_end']}? Assume a unique event of each type in the window."
        )
    if template_id == "B.2.1":
        word = {"first": "earliest", "last": "latest"}.get(p["position"], p["position"])
        return (
            f"Identify the {word} event of type {p['E']} between "
            f"{p['W_start']} and {p['W_end']}. If multiple events share the "
            f"extremal date, return all of them."
        )
    if template_id == "B.2.2":
        return (
            f"For firm {p['F']}, list all events of types [{', '.join(p['E_set'])}] between "
            f"{p['W_start']} and {p['W_end']}, ordered chronologically by anchor date."
        )
    if template_id == "B.3.1":
        return (
            f"Which firms had BOTH an event of type {p['E1']} AND an event of "
            f"type {p['E2']} between {p['W_start']} and {p['W_end']}? "
            f"Return distinct firm tickers."
        )
    if template_id == "B.3.2":
        return (
            f"Which firms played role {p['R']} in events of type {p['E']} "
            f"during BOTH the window [{p['W1_start']}, {p['W1_end']}] AND "
            f"the window [{p['W2_start']}, {p['W2_end']}]? Return distinct firm tickers."
        )
    if template_id == "B.3.3":
        return (
            f"Which firms played role {p['R']} in BOTH events of type {p['E1']} "
            f"AND events of type {p['E2']} between {p['W_start']} and "
            f"{p['W_end']}? Return distinct firm tickers."
        )
    if template_id == "B.4.1":
        return (
            f"Which firms had at least {p['N']} events of type {p['E']} between "
            f"{p['W_start']} and {p['W_end']}? Return distinct firm tickers."
        )
    if template_id == "B.4.2":
        return (
            f"Which firms played role {p['R']} in at least {p['N']} events of "
            f"type {p['E']} between {p['W_start']} and {p['W_end']}? "
            f"Return distinct firm tickers."
        )
    if template_id == "B.4.3":
        return (
            f"List events of type {p['E']} between {p['W_start']} and "
            f"{p['W_end']}, grouped by {p['bucket_unit']}. Attach a bucket label "
            f"to each row."
        )
    if template_id == "B.4.4":
        return (
            f"List all events of types [{', '.join(p['E_set'])}] between {p['W_start']} and "
            f"{p['W_end']}. Attach the event_type label to each row."
        )
    raise KeyError(f"Unknown template_id: {template_id!r}")
