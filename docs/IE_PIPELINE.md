# IE_PIPELINE.md — How the benchmark's ground truth was built

This document describes the **upstream pipeline** that derived QO-Bench's
operational event set $\widehat{E}$ (614 single-article-attestable events)
from the FNSPID news corpus and the S&P Capital IQ KeyDev event feed.

> **Distinction**: This pipeline produces the benchmark's *ground truth*.
> It is NOT the IE→SQL *baseline paradigm*. The IE→SQL baseline (one of
> the 5 deployable paradigms evaluated in the paper) lives at
> `../aggqa/baselines/ie_sql/`. They are two different uses of LLM-driven
> IE on the same corpus: one builds GT, one tries to *answer questions*
> using its own extraction.

## 1. What the GT pipeline does (3 stages)

### Stage A — Candidate matching

For each S&P KeyDev event in 2010–2023 (8 event types, ~16,414 events
total; we sample a 1,376-event subset to bound judging cost), find
candidate FNSPID articles by:

- **Exact ticker match** (FNSPID articles carry stock symbols; no
  fuzzy/alias normalization)
- **Per-event-type asymmetric date window** around the anchor date
  (windows are widened *after* anchor since news typically trails the
  legal/announcement date)

Date windows (from paper Table 6):

| Event type | Days before | Days after |
|---|---|---|
| M&A announcement | 7 | 14 |
| M&A completion | 7 | 90 |
| M&A cancellation | 7 | 30 |
| M&A rumor | 14 | 14 |
| CEO change | 14 | 60 |
| CFO change | 14 | 60 |
| IPO | 7 | 60 |
| Stock split | 7 | 14 |

Articles with empty bodies are dropped; duplicates removed by URL,
keeping the occurrence closest to the anchor.

This yields 25,888 candidate (event, article) pairs from the 1,376-event
subset.

### Stage B — 3-judge attestation

Three LLM judges independently label each (event, article) pair:

| Judge | Role |
|---|---|
| Gemma-4-31B-IT | Judge 1 |
| Qwen3.6-27B | Judge 2 (also the paper's answer model) |
| gpt-oss-120B | Judge 3 |

A pair is **attested** only on unanimous **3-of-3 confirmation**. Of the
25,888 candidate pairs, 1,591 (6.1%) are attested → yielding the
operational event set $\widehat{E}=614$ distinct single-article-attestable
events.

The script `../extraction/attestation_pipeline.py` is the Stage B runner.
Its hardcoded MongoDB URI and internal vLLM URL have been replaced with
`os.environ.get()` lookups; reviewers must set `MONGO_URI` and `LLM_URL`
to point at their own deployments.

### Stage C — Tuple normalization

Each attested event gets a typed tuple with:

- Firm (public identifier — ticker, CIK, or LEI; no proprietary IDs)
- Event type (one of 8 public-record-anchored types)
- Anchor date
- Role (Buyer / Target / Seller / executive role, depending on event type)
- Counterparty (optional, normalized)
- Provenance (FNSPID article IDs — evidence metadata only, not queryable)

The pipeline outputs a JSONL of tuples with provenance.

### Validation (paper §3.2)

A stratified sample of 221 accepted pairs was re-examined by 3 human
annotators (1 primary expert + 2 secondary). The 3-of-3 LLM consensus
achieves **94.3% precision** against expert labels; Cohen's κ=0.538
inter-annotator. Disagreements concentrate on M&A lifecycle ambiguity
(announcement vs. completion mentioned in passing).

## 2. What's in `../extraction/` (shipped)

- `attestation_pipeline.py` — the 3-judge attestation runner (Stage B / Stage C; secrets scrubbed).
- `README.md` — overview of the attestation pipeline.

The IE$\to$SQL baseline itself (schema generation, extraction worker, and
per-template SQL) lives in `../aggqa/baselines/ie_sql/`.

## 3. What's NOT shipped, and why

| Artifact | Size | Why not shipped |
|---|---|---|
| Full extraction output | ~88 MB | Permission-restricted on source; documented here for reviewers |
| Full raw KeyDev event table | ~3.5 GB | **Proprietary**. S&P license forbids redistribution. Reviewers need their own KeyDev subscription. |
| Full FNSPID corpus | ~28 GB | Public; fetch independently |
| Full-corpus attestation outputs (all 2010–2023 quarters) | varies | Out of scope; `attestation_pipeline.py` documents the format |

## 4. Reviewer reproduction path (if you want to re-derive ground truth)

1. **Get the data**:
   - FNSPID NASDAQ subset (public)
   - S&P Capital IQ KeyDev access (paid)
2. **Index FNSPID into MongoDB** with `event_id` cross-references to KeyDev.
3. **Stand up an LLM endpoint**: 3 different LLMs for judging (paper used
   Gemma-4-31B-IT + Qwen3.6-27B + gpt-oss-120B; substitutes likely work
   but agreement rates may differ).
4. **Set environment variables**:
   ```bash
   export MONGO_URI="mongodb://user:pass@your-mongo-host:27017/your_db"
   export LLM_URL="https://your-vllm-host:port/v1/chat/completions"
   ```
5. **Run**:
   ```bash
   python attestation_pipeline.py --start 2010-01-01 --end 2023-12-31
   ```
6. **Outputs** are persisted per-event under
   `<output_dir>/<event_type>/<event_id>.json`.

## 5. Relationship to IE→SQL baseline

The IE→SQL **baseline paradigm** (`../aggqa/baselines/ie_sql/`) is a
*separate* use of LLM-driven IE. It:

1. Generates a typed schema from event definitions (GPT-5.5),
2. Extracts tuples from articles into that schema (Qwen3.6-27B),
3. Answers questions via per-template SQL on a DuckDB.

It does **not** use the attestation pipeline's outputs. It runs IE
independently and is scored against the same gold answers as RAG / ReAct /
GraphRAG / LC-oracle. See `../aggqa/baselines/ie_sql/README.md`.
