# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Loads the upstream templates_config.json and builds response_format payloads
for OpenRouter's Structured Outputs feature.

Three enforcement modes, selected by config.STRUCTURED_OUTPUT_MODE:
  - "schema" : response_format=json_schema with strict=true  (sampling-time constraint)
  - "object" : response_format=json_object                    (valid JSON only)
  - "prompt" : returns None — caller relies on prompt + parse_model_output

Spec: §4.2 of 2026-05-12-rerank-and-eval-bundle-baselines-design.md
"""
from __future__ import annotations

import json
from functools import lru_cache

from qobench import config


@lru_cache(maxsize=1)
def load_templates_config() -> dict:
    """Load templates_config.json once. Keys: template id (e.g. "A.1.1")."""
    with open(config.TEMPLATES_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_response_format(template_id: str, mode: str = None) -> dict | None:
    """Build the OpenRouter response_format payload for one question.

    Args:
        template_id: e.g. "A.1.1" — must be a key in templates_config.json.
        mode: "schema" / "object" / "prompt". Defaults to config.STRUCTURED_OUTPUT_MODE.

    Returns:
        - dict payload for "schema" / "object"
        - None for "prompt" (caller doesn't set response_format)
    """
    if mode is None:
        mode = config.STRUCTURED_OUTPUT_MODE
    if mode == "prompt":
        return None
    if mode == "object":
        return {"type": "json_object"}
    if mode == "schema":
        cfg = load_templates_config()
        if template_id not in cfg:
            raise KeyError(f"Unknown template id: {template_id}")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "aggregate_qa_answer",
                "strict": True,
                "schema": cfg[template_id]["output_schema"],
            },
        }
    raise ValueError(f"Unknown STRUCTURED_OUTPUT_MODE: {mode!r}")
