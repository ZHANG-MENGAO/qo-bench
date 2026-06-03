# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""V2 bundle canonical url_hash whitelist.

Why: Milvus `fnspid_articles` contains ~282K articles total, but the v2
bundle (eval_bundle_v2_2026-05-14) defines only **22,984** of them as the
canonical substrate (corpus/articles.csv). All GT golden_chunks reference
URLs from those 22,984, so any retrieval hit OUTSIDE that set is wasted —
even if the model writes a correct-looking event from a non-canonical
article, it can never ground a `cited_url` against GT golden_chunks.

The canonical 22,984-article subset is a subset of the indexed collection.
Ideally the Milvus collection would be trimmed to exactly these articles, but
since we do not control the index server, we apply the filter client-side:
pass `url_hash in [...22,984 hashes...]` into every Milvus search expression.
Effect: retrieval only ever sees canonical articles.

Hash function: `sha256(article_url)[:32]` — matches how `fnspid_articles`
was indexed (verified 2026-05-14, 100/100 sanity check passed).

To regenerate the JSON whitelist file from `corpus/articles.csv`:
    python -m qobench.scripts.build_v2_whitelist

To disable the whitelist for an ablation run (substrate=full fnspid):
    export QOBENCH_DISABLE_V2_WHITELIST=1
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_WHITELIST_PATH = Path(__file__).parent / "v2_url_hash_whitelist.json"


@lru_cache(maxsize=1)
def load_v2_whitelist() -> frozenset[str] | None:
    """Return the 22,984 canonical url_hash strings, or None to skip filter.

    Returns None when:
      - QOBENCH_DISABLE_V2_WHITELIST=1 is set (explicit ablation)
      - the whitelist JSON file is missing (e.g. fresh checkout before the
        build script has been run)

    Callers should treat None as "no filter, fall back to full Milvus
    collection" — useful as a regression sentinel.
    """
    if os.environ.get("QOBENCH_DISABLE_V2_WHITELIST") == "1":
        return None
    if not _WHITELIST_PATH.exists():
        return None
    return frozenset(json.loads(_WHITELIST_PATH.read_text()))
