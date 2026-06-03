#!/usr/bin/env python3
# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-NC-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Inject event_definitions.md into GraphRAG system prompts.

Idempotent by default; pass --force to strip existing injection and re-inject
(needed when event_definitions.md was updated).
"""
import argparse
import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "prompts"
DEFS = Path(__file__).resolve().parents[3] / "benchmark" / "event_definitions.md"
MARKER = "<!-- INJECTED:event_definitions -->"
END_MARKER = "<!-- /INJECTED:event_definitions -->"
TARGETS = [
    "local_search_system_prompt.txt",
    "global_search_map_system_prompt.txt",
    "global_search_reduce_system_prompt.txt",
    "drift_search_system_prompt.txt",
]

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="Strip existing injection then re-inject.")
args = ap.parse_args()

defs_text = DEFS.read_text()
print(f"event_definitions: {len(defs_text)} chars")

section = (
    f"{MARKER}\n\n"
    "## Domain context — financial events\n\n"
    "You are answering questions about specific financial events for publicly traded firms.\n"
    "Event types and their precise definitions are below. Use these boundaries when extracting\n"
    "events from retrieved content (e.g. M&A_announce vs M&A_complete are distinct stages of\n"
    "the same deal — do not conflate them).\n\n"
    f"{defs_text}\n\n"
    f"{END_MARKER}\n\n"
    "---\n\n"
)

def strip_old_injection(text: str) -> str:
    """Remove old injection block. Handles both new (with END_MARKER) and old (start marker only) formats."""
    if END_MARKER in text:
        pat = re.compile(re.escape(MARKER) + r".*?" + re.escape(END_MARKER) + r"\s*-+\s*", re.DOTALL)
        return pat.sub("", text, count=1)
    if MARKER in text:
        # Old-style: drop from MARKER to the first lone "---" line (our delimiter)
        idx = text.find(MARKER)
        rest = text[idx:]
        m = re.search(r"\n---\n", rest)
        if m:
            return text[:idx] + rest[m.end():]
        return text[:idx]
    return text

for name in TARGETS:
    p = PROMPTS / name
    if not p.exists():
        print(f"  MISSING: {name}")
        continue
    content = p.read_text()
    if MARKER in content:
        if args.force:
            content = strip_old_injection(content)
            print(f"  STRIPPED old injection from {name}")
        else:
            print(f"  SKIP (already injected, use --force to refresh): {name}")
            continue
    p.write_text(section + content)
    print(f"  INJECTED: {name} (+{len(section)} chars)")
