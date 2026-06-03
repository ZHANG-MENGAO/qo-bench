# INFRA.md — External Service Expectations

This bundle is a research-code supplementary; it does NOT ship the
production-grade infrastructure required to fully re-run the experiments.
This document describes what services a reviewer would need to stand up
to reproduce the paper's numbers.

## Required services

### 1. Milvus 2.4+ (vector database, RAG / ReAct RAG)

The RAG and ReAct RAG baselines retrieve from a Milvus collection of news
article chunks.

- **Collection name**: `fnspid_articles` (configurable in `qobench/config.py`)
- **Vector field**: dense (Qwen3-Embedding-4B, 2560-dim), L2 distance
- **Sparse field**: BM25 (Milvus 2.4+ built-in)
- **Hybrid search**: RRF (`RRFRanker`, k=60) over sparse + dense
- **Group-by**: `url_hash` (deduplicates retrieved chunks per article)
- **Metadata fields**: `url_hash`, `date` (ISO string or yyyymmdd int —
  format discovered at runtime via `qobench/discover.py`), `ticker_list`

Indexing the ~5M chunks of the FNSPID NASDAQ corpus into this schema
requires:
- ~64 GB RAM on the Milvus host
- ~80 GB disk for the indexed collection
- ~2–4 hours of A100 GPU time to embed the corpus with Qwen3-Embedding-4B
- A re-implementation of the chunking + embedding pipeline (not shipped;
  see `IE_PIPELINE.md` for upstream sketch)

### 2. vLLM 0.17.1 (answer LLM serving — LLM/RAG paradigms + IE extraction)

The 3 local-vLLM LLM/RAG baselines (RAG, LC-oracle, no-context) and the
IE→SQL extraction stage run against a local vLLM serving Qwen3.6-27B:

```
vllm serve qwen-3.6-27b \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --served-model-name qwen-3.6-27b \
  --reasoning-parser qwen3
```

Requires 2× A100/H100 (80 GB each) for the model + FlashAttention-3 kernels.

### 3. vLLM 0.19.1 (GraphRAG — separate version)

GraphRAG (`qobench/baselines/graphrag/`) needs vLLM 0.19.1 (different version
because graphrag 3.0.9 expects newer vLLM API behavior). Serves both:
- chat: Qwen3.6-27B (port 8000)
- embed: Qwen3-Embedding-4B (port 8001)

### 4. OpenRouter (ReAct baseline only)

ReAct (`qobench/baselines/react.py`) routes through OpenRouter for
Qwen3.6-27B (provider routing not pinned). Reviewers would need an
OpenRouter API key; costs are roughly several USD for the 785-question set
across multi-round agentic search.

### 5. GPT-5.5 (IE→SQL Stage 1 — one-shot schema generation)

The IE→SQL paradigm's frozen schema (`qobench/baselines/ie_sql/schema_gen_output.md`)
was generated once by GPT-5.5 (T=0) from the event-definitions document
alone. The schema is shipped verbatim, so reviewers do **not** need GPT-5.5
to reproduce — they only need it if they want to regenerate the schema
from scratch.

### 6. GraphRAG conda environments

GraphRAG needs two conda envs:
- `graphrag` (~1.2 GB) — the Microsoft graphrag package + dependencies
- `vllm_server` (~11 GB) — vLLM 0.19.1 + Torch + CUDA wheels

GraphRAG indexing is expensive: ~16h on 4 GPUs with `extract_claims=true`.
The heavy index artifacts (parquets + lancedb, ~2.4 GB) live in
`<your-scratch>/graphrag-pilot/output/` at runtime — not shipped.

### 7. DuckDB (IE→SQL Stage 3 — local, no service)

The SQL execution stage runs on local DuckDB (no service needed). The
events DB is built once from the extracted tuples
(`qobench/baselines/ie_sql/build_events_db.py`).

### 8. Embedding backends (for queries)

Two backends supported via `QOBENCH_EMBED_BACKEND` env var (see
`qobench/infra/embedding.py`):
- `local` — Qwen3-Embedding-4B loaded on CPU (forced on macOS; MPS has a
  matmul shape bug). For Linux GPU hosts use GPU.
- `deepinfra` — DeepInfra hosted endpoint (cheaper for query workloads;
  recommended for reviewers without a local GPU).

### 9. Qwen3-Reranker-4B (RAG / ReAct reranker)

Used after Milvus hybrid retrieval to rerank candidates before passing
top-30 (RAG) / top-10 (ReAct, per round) to the answer LLM. Served via
DeepInfra in our runs.

## Discovery / smoke

`qobench/discover.py` is the Phase-0 probe: it connects to the configured
Milvus endpoint, lists collections, dumps the discovered schema, and
writes `outputs/config_runtime.json`. Running this is the first sanity
check after standing up the services above.

## Optional but useful

- `precache_retrieval.py` (`qobench/scripts/`) — pre-runs all retrievals
  into a JSONL cache, so the LLM-side runs become CPU/network-bound
  instead of Milvus-bound.
- `verify_embedding_parity.py` — sanity-checks that the local Qwen3
  embedding matches the DeepInfra-hosted embedding.

## Reproducibility cost summary

Stand-up cost for a reviewer:

| Service | Cost | Required for |
|---|---|---|
| Milvus host (64 GB RAM, 80 GB disk) | self-hosted | RAG, ReAct |
| FNSPID corpus + indexing (2–4h A100) | public download + GPU time | RAG, ReAct, GraphRAG |
| 2× A100/H100 for vLLM Qwen3.6-27B | GPU credits | All paradigms |
| OpenRouter API key + budget | ~$X-$XX | ReAct RAG only |
| DeepInfra (or own GPU) for reranker + embeddings | usage-based or self-hosted | RAG, ReAct |
| GPT-5.5 access (optional — schema is shipped) | API credits | IE→SQL schema regeneration only |
| S&P KeyDev license | paid | Ground-truth reconstruction (not needed to evaluate against shipped questions/answers) |

For a position paper, full reproduction is not expected — the supplementary
is intended for code/method inspection. The paper's numbers are reported
in its main results table.
