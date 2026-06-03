# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Single-load `dict[article_url → row]` for `eval_bundle_v3_2026-05-21/corpus/articles.csv`.

The corpus is byte-identical to v2_2026-05-18's articles.csv (same 22,984
articles, md5 14ed253a1ef2745664710d08dec1426f); v3 only changed attestation
and questions, not the substrate.

- `dtype=str` is required to preserve `event_ids_linked` as a JSON string
  (otherwise pandas tries to coerce the list literal into NaN/object soup).
- Single-process load, no concurrent build — pandas read_csv is single-threaded.
- ~128 MB CSV → ~250 MB Python dict; loads in ~3-5 s.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_CACHE: dict[str, dict] = {}


def load_articles_index(csv_path: str | Path) -> dict[str, dict]:
    """Load `articles.csv` once and return a `dict[article_url → row_dict]`.

    Subsequent calls with the same path return the cached dict.
    """
    key = str(Path(csv_path).resolve())
    if key in _CACHE:
        return _CACHE[key]

    df = pd.read_csv(csv_path, dtype=str)
    if "article_url" not in df.columns:
        raise ValueError(f"{csv_path} missing required column 'article_url'")
    df = df.fillna("")
    idx = {row["article_url"]: row.to_dict() for _, row in df.iterrows()}
    _CACHE[key] = idx
    return idx
