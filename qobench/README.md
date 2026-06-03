# qobench — Python package for the 5 baselines + shared infra

This Python package implements the **5 deployable baseline paradigms** that
the QO-Bench paper evaluates (RAG, ReAct RAG, LC-oracle, no-context,
GraphRAG, IE→SQL), plus their shared retrieval and scoring infrastructure.

*Note on naming*: `qobench` is the Python package codename; the published
benchmark is named **QO-Bench**. They refer to the same thing.

## Structure

| Subpackage | Paradigm / Purpose |
|---|---|
| `baselines/naive_rag.py` | **RAG** — single-shot hybrid retrieve (Qwen3-Embedding-4B dense + BM25, RRF, Qwen3-Reranker-4B) → top-30 chunks → Qwen3.6-27B answer. |
| `baselines/react.py` (+ `react_agent.py`, `notebook.py`) | **ReAct RAG** — LangChain `create_react_agent`, ≤5 retrieval calls per question, same hybrid stack. |
| `baselines/lc_oracle.py` | **LC-oracle ceiling** — feeds gold-attested chunks per question, no retrieval. Not a deployable paradigm; isolates retrieval contribution. |
| `baselines/no_context.py` | **no-context floor** — closed-book control with question only. |
| `baselines/graphrag/` | **GraphRAG** — microsoft/graphrag 3.0.9 (stock source), local + global + drift search modes. |
| `baselines/ie_sql/` | **IE→SQL** — schema-gen (GPT-5.5) → tuple extraction (Qwen3.6-27B) → per-template SQL on DuckDB. |
| `infra/retriever.py` | Milvus hybrid BM25 + dense retrieval with RRF + `group_by url_hash`. |
| `infra/embedding.py` | Qwen3-Embedding-4B wrapper (local + DeepInfra backends). |
| `infra/reranker.py` | Qwen3-Reranker-4B wrapper. |
| `infra/date_filter.py` | Milvus filter-expression builder for date windows. |
| `infra/{golden_chunks,benchmark_loader,question_params,question_renderer,...}` | Benchmark / question handling. |
| `eval/eval.py` | Strict scorer (paper's primary metric is the tolerant variant). |
| `eval/eval_tolerant.py` | Tolerant scorer (±7-day date tolerance — the primary metric in the paper). |
| `prompts/prompt_template.py` | System prompt + user-prompt builder + LLM-output parser shared by the 4 LLM/RAG baselines. |
| `prompts/eval_reference.py` | Original (un-patched) scorer kept for diff reference. |
| `scripts/` | Reproducibility helpers — pre-cache retrieval, verify embedding parity. |

## Requirements

See `../requirements.txt` at bundle root. The 4 LLM/RAG baselines and
GraphRAG share a vLLM-hosted Qwen3.6-27B; see `../docs/INFRA.md` for the
full external-service expectations.

## Running

This package requires external services (Milvus, vLLM, optionally OpenRouter
for ReAct) to do useful work. Reading the code does not. Just open any
baseline file in `baselines/` and follow imports into `infra/`, `eval/`,
`prompts/`.
