# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-NC-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
# qobench/discover.py
"""Phase 0: probe Milvus, dump schema, write runtime config.

Halts with a diagnostic message if:
  - TCP port 19530 unreachable
  - No collections found
  - Schema lacks url_hash or any plausible date field
  - Vector dim doesn't match a known Qwen3 embedding size
"""
from __future__ import annotations

import json
import socket
import sys
from urllib.parse import urlparse

from pymilvus import DataType, MilvusClient

from qobench import config

# Known Qwen3 embedding dim → model mapping
QWEN3_EMBED_DIM_TO_MODEL = {
    1024: "Qwen/Qwen3-Embedding-0.6B",
    2560: "Qwen/Qwen3-Embedding-4B",
    4096: "Qwen/Qwen3-Embedding-8B",
}

# Plausible date field names (in priority order)
DATE_FIELD_CANDIDATES = ["anchor_date", "article_date", "date", "event_date", "published_date"]


def _field_type_code(f: dict) -> int | None:
    """Milvus describe_collection returns field type either as a DataType enum
    or as an int code. Normalize to int."""
    t = f.get("type")
    if t is None:
        return None
    if isinstance(t, DataType):
        return int(t)
    try:
        return int(t)
    except (TypeError, ValueError):
        return None


def check_tcp_reachable(uri: str, timeout_s: int = 5) -> bool:
    parsed = urlparse(uri)
    host = parsed.hostname
    port = parsed.port or 19530
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (OSError, socket.timeout):
        return False


def halt_with_message(msg: str) -> None:
    print(f"\n{'='*60}\nDISCOVERY HALTED\n{'='*60}\n{msg}\n")
    sys.exit(1)


def main() -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. TCP reachability
    print(f"[1/4] Probing TCP {config.MILVUS_URI} ...")
    if not check_tcp_reachable(config.MILVUS_URI):
        halt_with_message(
            f"Cannot reach {config.MILVUS_URI}.\n"
            f"Likely cause: the Milvus host is on a private subnet not routable\n"
            f"from this machine. Check:\n"
            f"  - VPN / SSH-tunnel / jumphost requirements\n"
            f"  - Whether your IP is in the host's allowlist\n"
            f"  - Or whether discovery must run from inside the network where the index lives"
        )
    print("    ok")

    # 2. Connect + list collections
    print(f"[2/4] Connecting and listing collections ...")
    try:
        client = MilvusClient(
            uri=config.MILVUS_URI,
            token=config.MILVUS_TOKEN,
            db_name=config.MILVUS_DB_NAME,
        )
        collections = client.list_collections()
    except Exception as e:
        halt_with_message(
            f"Connected to {config.MILVUS_URI} but Milvus rejected the request:\n"
            f"  {e}\n\n"
            f"Check:\n"
            f"  - Whether the read-only token is still valid\n"
            f"  - Token format (user:password vs Bearer token)\n"
            f"  - Whether db_name='{config.MILVUS_DB_NAME}' is correct\n"
            f"  - Or whether discovery must run from inside the network where the index lives"
        )
    if not collections:
        halt_with_message(f"No collections in db '{config.MILVUS_DB_NAME}'. Check the Milvus configuration.")
    print(f"    found: {collections}")

    # 3. Pick the most likely collection
    # Strategy: if more than one, prefer one with 'fnspid' / 'temporal' / 'event' in name
    chosen = collections[0]
    for c in collections:
        low = c.lower()
        if any(tok in low for tok in ("fnspid", "temporal", "event", "v2", "agg")):
            chosen = c
            break
    print(f"    chosen collection: {chosen}")

    # 4. Describe + sample one row
    print(f"[3/4] Describing schema for '{chosen}' ...")
    desc = client.describe_collection(chosen)
    fields = desc.get("fields", [])
    field_summary = [{
        "name": f.get("name"),
        "type": str(f.get("type")),
        "is_primary": f.get("is_primary", False),
        "params": f.get("params", {}),
    } for f in fields]

    # Try to identify vector / sparse / date / url_hash fields
    vector_field = None
    sparse_field = None
    date_field = None
    date_field_type = None  # int code: 5=INT64, 21=VARCHAR
    url_hash_field = None
    vector_dim = None
    for f in fields:
        name = f.get("name", "")
        tcode = _field_type_code(f)
        if tcode == int(DataType.FLOAT_VECTOR):
            vector_field = name
            vector_dim = f.get("params", {}).get("dim")
        elif tcode == int(DataType.SPARSE_FLOAT_VECTOR):
            sparse_field = name
        if name == "url_hash":
            url_hash_field = name
    for cand in DATE_FIELD_CANDIDATES:
        for f in fields:
            if f.get("name") == cand:
                date_field = cand
                date_field_type = _field_type_code(f)
                break
        if date_field:
            break

    # Sample one row to confirm date format
    print(f"[4/4] Sampling 1 row for sanity ...")
    try:
        sample = client.query(
            collection_name=chosen,
            filter="",
            output_fields=[f.get("name") for f in fields[:8]],
            limit=1,
        )
    except Exception as e:
        sample = [{"_query_error": str(e)}]

    # Identify embedding model from vector dim
    embed_model = QWEN3_EMBED_DIM_TO_MODEL.get(vector_dim, f"UNKNOWN_DIM_{vector_dim}")

    # Derive date format from field type. INT64 (code 5) means YYYYMMDD integer;
    # VARCHAR (code 21) means ISO-string. Other types are unsupported.
    if date_field_type == int(DataType.INT64):
        date_format = "yyyymmdd_int"
    elif date_field_type == int(DataType.VARCHAR):
        date_format = "iso_string"
    else:
        date_format = f"unknown_type_{date_field_type}"

    # Write outputs
    schema_dump = {
        "collection_name": chosen,
        "all_collections": collections,
        "fields": field_summary,
        "sample_row": sample,
    }
    config.SCHEMA_PATH.write_text(json.dumps(schema_dump, indent=2, default=str))
    print(f"    wrote {config.SCHEMA_PATH}")

    runtime = {
        "collection_name": chosen,
        "vector_field": vector_field,
        "vector_dim": vector_dim,
        "sparse_field": sparse_field,
        "date_field": date_field,
        "date_format": date_format,
        "url_hash_field": url_hash_field,
        "embed_model_guess": embed_model,
    }
    config.RUNTIME_PATH.write_text(json.dumps(runtime, indent=2))
    print(f"    wrote {config.RUNTIME_PATH}")

    # Validate
    issues = []
    if not vector_field:
        issues.append("no dense vector field detected")
    if not url_hash_field:
        issues.append(f"no url_hash field (candidates: {[f['name'] for f in field_summary]})")
    if not date_field:
        issues.append(f"no date field among {DATE_FIELD_CANDIDATES}")
    if vector_dim and vector_dim not in QWEN3_EMBED_DIM_TO_MODEL:
        issues.append(f"vector dim {vector_dim} doesn't match a known Qwen3 size")

    if issues:
        halt_with_message(
            "Schema validation failed:\n  - " + "\n  - ".join(issues) +
            f"\n\nFull schema dump: {config.SCHEMA_PATH}"
        )

    print(f"\n[OK] Discovery complete.")
    print(f"     collection      = {chosen}")
    print(f"     vector          = {vector_field} (dim={vector_dim})")
    print(f"     sparse (BM25)   = {sparse_field}")
    print(f"     date            = {date_field}")
    print(f"     url_hash        = {url_hash_field}")
    print(f"     embed model     = {embed_model}")


if __name__ == "__main__":
    main()
