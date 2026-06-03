# BASELINES.md — Implementation Details for the Baseline Paradigms

This document describes the implementations evaluated in the QO-Bench paper:
**5 deployable paradigms** (RAG, ReAct RAG, GraphRAG local, GraphRAG global,
IE→SQL) plus a **long-context oracle ceiling** and a **no-context floor**
for control. Holding the answer LLM (Qwen3.6-27B) fixed across all
paradigms, the gap from each deployable to the LC-oracle ceiling isolates
its retrieval contribution.

---

## §A — Shared answer LLM and decoding

All deployable paradigms and the LC-oracle ceiling use the same answer
model: **Qwen3.6-27B**.

| Paradigm   | Answer-LLM backend                                |
|------------|---------------------------------------------------|
| LC-oracle  | Local vLLM 0.17.1 (2-GPU TP, bfloat16, FA-3)      |
| no-context | Same vLLM 0.17.1 server                           |
| RAG        | Same vLLM 0.17.1 server                           |
| ReAct RAG  | OpenRouter (`qwen/qwen3.6-27b`, provider unpinned) |
| GraphRAG   | Local vLLM 0.19.1 (different version for graphrag compat) |
| IE→SQL     | Local vLLM 0.17.1 (extraction stage only; SQL stage has no LLM) |

vLLM 0.17.1 flags (the 3 local-vLLM LLM/RAG paradigms):
`--tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.90
--served-model-name qwen-3.6-27b --reasoning-parser qwen3`

**Shared decoding**: `temperature = 0.0` (greedy), request timeout 600s,
max retries 3.

**Thinking mode**: ON for all paradigms (Qwen3 chat-template default).
For the local-vLLM paradigms, `--reasoning-parser qwen3` routes the
`<think>…</think>` block into `reasoning_content` and leaves `content`
clean for the JSON answer. For ReAct on OpenRouter, thinking depth is
controlled via `reasoning.effort = medium`.

**Output token budgets (per paradigm)**:

| Paradigm    | Budget       | Reason                                                                                       |
|-------------|--------------|----------------------------------------------------------------------------------------------|
| LC-oracle   | **61,120**   | = `max_model_len` 262,144 − input budget 200,000 − safety 1,024                              |
| no-context  | **16,384**   | Closed-book questions are short; capping prevents runaway generation                         |
| RAG         | vLLM default | Single-shot, bounded by `max_model_len` − input                                              |
| ReAct RAG   | controlled via `reasoning.effort` | OpenRouter handles routing; effort = medium |
| GraphRAG    | graphrag 3.0.9 defaults | See `aggqa/baselines/graphrag/settings.yaml` |
| IE→SQL (extraction) | 8,192 | Per-article extraction is bounded; multi-chunk articles get multiple calls |

---

## §B — RAG (`aggqa/baselines/naive_rag.py`)

Single-shot retrieve → prompt → LLM.

| Component | Value |
|---|---|
| Retrieval | Milvus hybrid: dense (Qwen3-Embedding-4B, 2560-dim) + BM25 (Milvus 2.4+ built-in) |
| Fusion | RRF (`RRFRanker`, k=60), group_by `url_hash` |
| Reranker | Qwen3-Reranker-4B via DeepInfra |
| Top-k | 100 candidates from hybrid → 30 reranked → 30 to LLM |
| Filter | Per-question date window (event-type-asymmetric — see paper Table 6) + optional firm ticker pre-filter |
| Prompt | `aggqa/prompts/prompt_template.py` (shared across LLM/RAG paradigms; event-definitions prepended) |

---

## §C — ReAct RAG (`aggqa/baselines/react.py`)

Agentic, multi-round retrieval. LangChain `langchain.agents.create_agent`
(v1.0, replaces deprecated `langgraph.prebuilt.create_react_agent`).

| Component | Value |
|---|---|
| Tool | Single `search_news` tool wrapping the hybrid retriever |
| Per-round k | 50 hybrid → 10 reranked → 10 to LLM |
| Max rounds | 5 retrieval calls per question (`recursion_limit=25` total internal LangGraph steps) |
| Per-question timeout | 900s |
| Retriever | Fresh `MilvusHybridRetriever` instance per question (avoids shared mutable state across async coroutines) |
| Prompt | Single uniform prompt for all templates (no per-template hints — fairness baseline) |

---

## §D — LC-oracle ceiling (`aggqa/baselines/lc_oracle.py`)

Not a deployable paradigm — feeds each question its gold-attested chunks
directly (linked via provenance). Isolates retrieval failure: any gap
between LC-oracle and a deployable paradigm is the deployable's retrieval
contribution.

| Component | Value |
|---|---|
| Context | All 3-of-3 attested chunks for each event contributing to the gold answer |
| Input budget | 200,000 tokens (rest reserved for reasoning + output) |
| Truncation | None — paper §3 mandates non-truncation for ceiling honesty |
| Reasoning effort | Full Qwen3 reasoning |

---

## §E — no-context floor (`aggqa/baselines/no_context.py`)

Closed-book control: question only, no retrieval, no context. Documents
how much the LLM "knows" about the specific events without evidence.

| Component | Value |
|---|---|
| Context | None |
| Prompt | Closed-book system prompt + question; same answer-schema discipline as RAG |
| Decoding | temperature=0, thinking ON, max_tokens 16,384 |

The paper reports a 0.6% overall on this floor — most aggregate questions
require evidence that the answer LLM cannot produce from training-time
knowledge alone.

---

## §F — GraphRAG (`aggqa/baselines/graphrag/`)

Microsoft graphrag 3.0.9, **source code unmodified**. Both local and global
search modes evaluated.

| Component | Value |
|---|---|
| Library | `graphrag==3.0.9` |
| Chat backend | Qwen3.6-27B via vLLM 0.19.1 |
| Embed backend | Qwen3-Embedding-4B via vLLM 0.19.1 |
| Chunking | 1200 tokens / 100 overlap / `o200k_base` |
| Index settings | `extract_claims=true`, `extract_graph.max_gleanings=2`, `summarize_descriptions.max_length=1000`, `prune_graph.min_node_freq=1`, `concurrent_requests=128` |
| Search modes | local (entity match → neighborhood) + global (map-reduce over community reports) |
| Query settings | graphrag 3.0.9 stock defaults; `top_k_mapped_entities=10`, `community_level=2` |
| Prompts | 5 of 12 user-facing prompts modified for (a) event-ontology prepend and (b) URL-aware citation; stock 7 of 12 not bundled (see microsoft/graphrag upstream) |

See `aggqa/baselines/graphrag/README.md` for full details. Note: GraphRAG
indexing is expensive (~16h on 4 GPUs with `extract_claims=true`); the
heavy index artifacts (parquets + lancedb, ~2.4 GB) are not bundled.

---

## §G — IE→SQL (`aggqa/baselines/ie_sql/`)

Three-stage paradigm:

1. **Schema generation** — GPT-5.5 (T=0) drafts a typed schema from
   `event_definitions.md` alone, never seeing question templates. Output
   frozen verbatim, zero manual corrections.
2. **Extraction** — Qwen3.6-27B (T=0, reasoning ON) extracts event tuples
   from each FNSPID article into the frozen schema. Output: per-article
   JSON with extracted tuples + provenance.
3. **SQL execution** — per-template SQL skeletons dispatch each question
   to a SQL family against a DuckDB events DB built from the extracted
   tuples. **No LLM at query time**.

| Component | Value |
|---|---|
| Schema generator | GPT-5.5, T=0 |
| Extractor | Qwen3.6-27B via vLLM 0.17.1, T=0, reasoning ON |
| SQL engine | DuckDB |
| Coordination | Filesystem-claim locks for multi-worker extraction |
| Per-template SQL | SQL families in `run_ie_sql.py:DISPATCH` covering all 18 paper templates (A/B IDs) |

See `aggqa/baselines/ie_sql/README.md` for runbook. IE→SQL is the
strongest deployable paradigm on QO-Bench's compositional operators
(intersection, count, group), where SQL executes natively. Its main
weakness is cross-event temporal joins where the extractor under-populates
`cluster_key_hint` (B.1.5 announce↔complete linking).

---

## Paper main results (paper Table 3 — strict ±7d)

| Operator family | LC-oracle | RAG | ReAct RAG | GR-local | GR-global | IE→SQL |
|---|---|---|---|---|---|---|
| Filter / project | 77.6 | 53.6 | 55.4 | 3.5 | — | 50.6 |
| Role / type | 35.3 | 18.2 | 51.1 | 0.0 | — | 55.6 |
| Cap A (all) | 67.0 | 44.8 | 54.3 | 2.6 | — | 51.9 |
| Temporal join | 52.3 | 11.1 | 15.7 | 0.3 | — | 21.1 |
| Ordering | 59.9 | 29.8 | 27.8 | 0.5 | — | 17.8 |
| Intersection | 3.9 | 6.0 | 9.5 | 0.4 | — | 50.9 |
| Count / group | 56.2 | 30.0 | 3.4 | 0.3 | — | 51.2 |
| Cap B (all) | 47.1 | 18.5 | 13.5 | 0.3 | — | 33.1 |
| **Overall** | **52.2** | **25.2** | **23.9** | **0.9** | — | **37.9** |

Numbers reproduce Table 3 of the paper. GR-global pending its full run at
paper submission.

Three observations from the paper:
1. **Even the ceiling is operator-bound**: LC-oracle scores 77.6% on
   filtering but only 3.9% on intersection. Operator *execution* — not
   only retrieval — is a bottleneck.
2. **Filtering is broadly solved**: RAG, ReAct RAG, IE→SQL all recover
   about half the ceiling on filter/project; GR-local is the exception.
3. **Ranking inverts on composition**: IE→SQL dominates intersection
   (50.9% vs. RAG/ReAct 6.0/9.5) and counting/grouping (51.2 vs. ReAct
   3.4) — SQL executes these natively while a generator must reconstruct
   from prose.
