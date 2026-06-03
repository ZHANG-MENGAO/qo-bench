# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
# qobench/config.py
"""Single source of truth for all knobs in the QO-Bench naive RAG run.

Values come from spec §4 (2026-05-05). Some fields are unknown until
discover.py resolves them; those are marked None and must be set in
config_runtime.json (written by discover.py, read by retriever.py / run.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env eagerly so module-level constants can read from it at import time.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# -------- Static config (locked in spec §4) --------

# Milvus
MILVUS_URI = os.environ.get("MILVUS_URI", "http://<your-milvus-host>:19530")
# Read token for the Milvus collection of FNSPID news chunks. The codebase
# is restricted to read-only operations (search / query / describe_collection
# / list_collections); never insert / delete / drop / upsert.
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
if not MILVUS_TOKEN:
    raise RuntimeError(
        "MILVUS_TOKEN missing from environment. Set MILVUS_TOKEN=<your-token> "
        "in your shell or a .env file at the bundle root."
    )
MILVUS_DB_NAME = "default"
TOP_K_INITIAL = 100   # Milvus hybrid search returns this many candidates (Naive RAG)
TOP_K_RERANKED = 30   # Reranker keeps this many — same K reaches the LLM as before (Naive RAG)

# ReAct uses a TIGHTER per-round K than Naive RAG:
# multi-round + cross-round dedup compensates — over 4 rounds ReAct sees up to
# 10 × 4 = 40 unique chunks, still more than Naive's 30. Per-round LLM context
# is ~1/3 the size, freeing reasoning budget and reducing attention dilution.
# Reranker selectivity goes from 1:3.33 (100→30) to 1:5 (50→10); Qwen3-Reranker
# on top-50 RRF candidates still surfaces the relevant chunks.
TOP_K_INITIAL_REACT = 50
TOP_K_RERANKED_REACT = 10

GROUP_BY_FIELD = "url_hash"
DATE_WINDOW_PAD_DAYS = 30
RRF_K = 60

# Reranker (DeepInfra Qwen3-Reranker-4B)
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/inference"
RERANK_MODEL = "Qwen/Qwen3-Reranker-4B"
RERANK_TIMEOUT_S = 90   # under concurrency, 30s can be too tight
# DeepInfra occasionally resets the connection mid-handshake (errno 54
# "Connection reset by peer"). requests has no built-in retry, so we wrap
# with bounded retry on connection/timeout errors only — 4xx/5xx still
# propagate immediately because those are application-level signals
# (rate limit, server bug) that masking with retry would just hide.
RERANK_RETRY_MAX_ATTEMPTS = 3   # 1 initial + 2 retries
RERANK_RETRY_BACKOFF_S = 1.5    # linear scaling: 1.5s, 3.0s

# Structured Outputs enforcement level
# "schema" → response_format=json_schema strict=true (default)
# "object" → response_format=json_object (looser)
# "prompt" → no response_format; rely on prompt + parse_model_output (fallback)
STRUCTURED_OUTPUT_MODE = "schema"

# LLM (OpenRouter)
# No max_tokens cap: Qwen3.6-27b has a 256K context window, and a 30-chunk
# retrieval window can be ~30k input tokens; reasoning + output combined may
# need significant budget. Letting the provider's default upper bound apply
# is safer than hard-coding a number that quietly truncates.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "qwen/qwen3.6-27b"
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_S = 600
LLM_RETRY_ATTEMPTS = 3

# ReAct search-tool hard budget (per question). The 6th attempt is soft-blocked
# inside make_search_tool — the LLM sees a BUDGET_EXHAUSTED string and finalizes.
# Decoupled from RECURSION_LIMIT (which counts all node visits in the LangGraph
# super-step sense), which stays as a 30-step catch-all fail-safe.
REACT_SEARCH_BUDGET = 5

# LC-oracle budget.
# Qwen3.6-27B native max_position_embeddings is 262144 (no YaRN, verified from HF config).
# We hard-cap input at 200K to leave a fixed 62K reserve for <think>+answer, decoupling
# input/output budgeting from max-model-len budget races. Oversize questions hard-skip
# to errors.jsonl as DNF; predictions.json denominator = 763 / 768 on cap50.
LC_ORACLE_MAX_MODEL_LEN  = 262_144
LC_ORACLE_INPUT_BUDGET   = 200_000
LC_ORACLE_OUTPUT_RESERVE = LC_ORACLE_MAX_MODEL_LEN - LC_ORACLE_INPUT_BUDGET   # = 62_144

# Paths (PROJECT_ROOT is defined at the top of this file for early .env loading)
HANDOVER_DIR = Path(__file__).parent / "prompts"  # in-package shared prompt scaffold
# Canonical evaluation set: 785 questions (200 Cap A + 585 Cap B) across the
# 18 paper templates, cap=50 stratified. Built from the 3-of-3 attestation
# pipeline documented in docs/IE_PIPELINE.md. Pass --benchmark to override.
BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "questions" / "questions.jsonl.gz"
TEMPLATES_CONFIG_PATH = PROJECT_ROOT / "benchmark" / "templates_config.json"
OUTPUTS_DIR = Path(__file__).resolve().parent / "runs" / "outputs"
SCHEMA_PATH = OUTPUTS_DIR / "discovered_schema.json"
RUNTIME_PATH = OUTPUTS_DIR / "config_runtime.json"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.jsonl"
RESULTS_PATH = OUTPUTS_DIR / "results.json"
RUN_LOG_PATH = OUTPUTS_DIR / "run_log.txt"
RUN_NOTES_PATH = OUTPUTS_DIR / "RUN_NOTES.md"

# -------- Runtime config (set by discover.py) --------

def load_runtime() -> dict:
    """Load runtime-resolved fields. Raises if discover.py hasn't run."""
    if not RUNTIME_PATH.exists():
        raise RuntimeError(
            f"Runtime config missing: {RUNTIME_PATH}\n"
            "Run `python -m qobench.discover` first."
        )
    return json.loads(RUNTIME_PATH.read_text())


def load_openrouter_key() -> str:
    # .env is already loaded at module import; this load_dotenv is idempotent.
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing from .env")
    return key


def load_deepinfra_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError("DEEPINFRA_API_KEY missing from .env")
    return key
