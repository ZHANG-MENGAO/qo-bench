# IE→SQL Baseline

The information-extraction-to-SQL paradigm (paper §4 "IE→SQL"). It runs in
three stages:

1. **Schema generation** — an LLM (GPT-5.5, T=0) drafts a typed database
   schema from the shared event-definitions document alone (never sees the
   question templates). The output is frozen verbatim with zero manual
   corrections.
2. **Extraction** — a separate LLM (Qwen3.6-27B, T=0, reasoning mode)
   extracts event tuples from each article into the frozen schema.
3. **SQL execution** — per-template SQL skeletons translate questions to
   SQL executed against the events database. No LLM at query time.

Stage 1 is the schema-leakage control: the IE→SQL paradigm is evaluated as
an intrinsic test of schemaful retrieval, not as a hand-tuned upper bound.

## Files

| File | Purpose |
|---|---|
| `schema_gen_prompt.md` | Stage 1 — system prompt fed to GPT-5.5 |
| `schema_gen_output.md` | Stage 1 — raw GPT-5.5 output, frozen |
| `schema_v1_frozen.md` | Stage 1 — the canonical schema, TypeScript form |
| `extraction_prompt.md` | Stage 2 — extraction system prompt template (uses `{{EVENT_DEFINITIONS_MD}}` and `{{SCHEMA_TS}}` placeholders) |
| `event_definitions.md` | Stage 2 — shared task ontology (same file fed to all paradigms) |
| `extraction_worker.py` | Stage 2 — cooperative multi-worker extraction runner; reads articles JSONL, writes per-article extracted JSONs |
| `build_events_db.py` | Stage 2.5 — builds a DuckDB events DB from the extracted JSONs (multi-stage entity/event canonicalization via union-find) |
| `run_ie_sql.py` | Stage 3 — the SQL paradigm runner; dispatches each question to a per-template SQL family and writes a predictions JSONL ready for `../eval/eval.py` |

## How to run

### Stage 2 (extraction)

Requires vLLM serving Qwen3.6-27B on `$VLLM_ENDPOINT` (default
`http://localhost:8000/v1/chat/completions`).

```bash
mkdir -p claims results logs
python3 extraction_worker.py \
  --input input_articles.jsonl \
  --claims-dir claims \
  --results-dir results \
  --worker-id ${PBS_JOBID:-local-1}
```

For parallelism, launch multiple PBS/SLURM jobs; coordination is via
filesystem-claim locks (`claims/<article_id>.claim`).

### Stage 2.5 (build DB)

```bash
python3 build_events_db.py \
  --extracted-dir results/ \
  --db-out events.duckdb
```

### Stage 3 (SQL run)

```bash
python3 run_ie_sql.py \
  --db events.duckdb \
  --questions ../../../benchmark/questions/questions.jsonl.gz \
  --output predictions.jsonl
```

Then score with `../../eval/eval.py` (or `eval_tolerant.py`).

## Per-template SQL coverage

`run_ie_sql.py:DISPATCH` covers all 18 templates, keyed by the paper's
A/B template IDs (matching `benchmark/questions/questions.jsonl.gz`).

## Paper results

IE→SQL achieves the strongest overall score among deployable paradigms on
the cap50 benchmark (paper Table 3): **37.9% overall recall** under the
paper's primary metric (tolerant ±7-day, covered subset), with particular
strength on intersection and counting/grouping operators where SQL executes
natively.

## Models used

- **Schema generator**: GPT-5.5 (T=0)
- **Extractor**: Qwen3.6-27B served via vLLM (T=0, reasoning mode)
- **SQL engine**: DuckDB (no LLM at query time)
