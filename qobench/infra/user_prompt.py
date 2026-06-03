# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""User-message assembly for naive RAG / ReAct / LC-oracle.

Replaces handover's `prompt_template.build_user_prompt` at the call sites.
Two reasons for our own implementation rather than monkey-patching the
handover constants:

  1. The handover's `PUBLIC_EVENT_ONTOLOGY` (5/5 vintage) lists 13 event
     names including Chapter_11_*, DIP_financing, restructuring, emergence,
     liquidation, plus split CEO_change_announce/effective and IPO_pricing.
     The authoritative event_definitions.md (5/14) uses 8 canonical names
     and is what cap50 GT uses. The handover list is stale.

  2. The handover's USER_PROMPT_TMPL has no slot for per-template output
     schema, and we now want one (see `output_schemas.py`).

We still reuse handover's `format_chunk` for individual chunk rendering —
that part is template-agnostic and was already vetted by Naive RAG / ReAct.

Source of truth for schemas: eval_bundle_v2_2026-05-18/schema.md
"""
from __future__ import annotations

import sys
from typing import Iterable

from qobench import config
from qobench.infra.output_schemas import get_schema_for_template
from qobench.infra.question_renderer import render_question

# Reuse handover's per-chunk formatter (template-agnostic, vetted).
from qobench.prompts.prompt_template import format_chunk  # noqa: E402


def build_user_message(
    question: dict,
    chunks: Iterable[dict],
    *,
    max_body_chars: int = 100_000,
) -> str:
    """Build the user-role message for one question.

    Structure:
        # Articles (n={k})

        [1] <date> | <ticker> | <title>
        <body>

        [2] ...

        # Output schema

        <per-template schema block>

        # Question

        <nl_question verbatim>

    Arguments:
        question: bundle question record (must have `id`, `nl_question`).
        chunks: per-chunk dicts {date, ticker, title, body} — formatter is
            tolerant of missing fields per handover's format_chunk.
        max_body_chars: per-chunk body truncation. LC-oracle uses 100K
            (effectively no truncation; the budget gate is upstream in
            golden_chunks.build_oracle_context). Naive RAG / ReAct may use
            the handover default of 1500.
    """
    chunks_list = list(chunks)
    k = len(chunks_list)

    if chunks_list:
        articles_block = "\n\n".join(
            format_chunk(i, c, max_body_chars=max_body_chars)
            for i, c in enumerate(chunks_list, start=1)
        )
    else:
        articles_block = ""

    schema_block = get_schema_for_template(question["id"], question.get("legacy_id"))
    nl = render_question(question)

    return (
        f"# Articles (n={k})\n\n"
        f"{articles_block}\n\n"
        f"{schema_block}\n"
        f"# Question\n\n"
        f"{nl}"
    )
