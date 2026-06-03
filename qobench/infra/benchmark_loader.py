# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Multi-format benchmark loader with optional per-template question cap.

Supports JSON (with `questions` key or bare list), JSONL, and gzipped JSONL.
The per-template cap lets one runner CLI cover everything from a 21-q smoke
(cap=1 over 21 templates) to a 11,822-q full run (no cap).
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path


def load_benchmark(
    path: Path,
    per_template_cap: int | None = None,
) -> list[dict]:
    """Load a benchmark file and return a flat list of question dicts.

    Format detected by extension:
      - .json         → expects {"questions": [...]} or top-level list
      - .jsonl        → one question per line
      - .jsonl.gz     → gzipped JSONL

    If per_template_cap is set, keep at most N questions per template_id
    (`q["id"]`), selected by stable sort on qid (deterministic).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    questions = _read_records(path)

    if per_template_cap is not None:
        questions = _apply_per_template_cap(questions, per_template_cap)

    return questions


def _read_records(path: Path) -> list[dict]:
    """Dispatch to the right reader based on extension."""
    name = path.name
    if name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return _read_jsonl_lines(f)
    if name.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as f:
            return _read_jsonl_lines(f)
    if name.endswith(".json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return list(raw.get("questions", []))
        if isinstance(raw, list):
            return raw
        raise ValueError(
            f"JSON benchmark must be a list or {{'questions': [...]}}, got {type(raw).__name__}"
        )
    raise ValueError(
        f"Unsupported benchmark extension on {path.name!r}; "
        f"expected .json / .jsonl / .jsonl.gz"
    )


def _read_jsonl_lines(fp) -> list[dict]:
    out = []
    for line in fp:
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _apply_per_template_cap(questions: list[dict], cap: int) -> list[dict]:
    """Group by `q["id"]`, sort each group by qid, take first `cap`,
    concatenate groups in alphabetical template-id order."""
    by_template: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_template[q["id"]].append(q)

    out: list[dict] = []
    for tid in sorted(by_template.keys()):
        group = sorted(by_template[tid], key=lambda q: q["qid"])
        out.extend(group[:cap])
    return out
