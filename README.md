# QO-Bench

**Paper**: *QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples*

**Authors**: Mengao Zhang, Xiang Yang*, Liu Chang, Tan Tianhui, Ke-Wei Huang
Asian Institute of Digital Finance, National University of Singapore.
Correspondence: Mengao Zhang (`mengaoz@nus.edu.sg`).
*Work done during an internship at the Asian Institute of Digital Finance, National University of Singapore.

**Repository**: https://github.com/ZHANG-MENGAO/qo-bench

This repository accompanies the paper. It contains the **QO-Bench**
benchmark, the 5 deployable baseline paradigms, the long-context oracle
ceiling, the IE pipeline that derives the benchmark's ground truth, and
supporting documentation.

## What's in this bundle

```
emnlp_supplementary/
├── README.md                          (this file)
├── LICENSE                            (MIT — code)
├── DATA_LICENSE                       (CC-BY-4.0 — benchmark questions + derived data)
├── RESPONSIBLE_NLP_CHECKLIST.md       (full ARR checklist response)
├── requirements.txt                   (pinned Python deps)
│
├── qobench/                           (the QO-Bench Python package)
│   ├── baselines/                     (5 deployable paradigms + LC-oracle ceiling + no-context floor)
│   │   ├── naive_rag.py               (RAG: hybrid dense+BM25 retrieval, top-30 reranked, Qwen3.6-27B answers)
│   │   ├── react.py + react_agent.py + notebook.py  (ReAct RAG: agentic, ≤5 retrieval calls per question)
│   │   ├── lc_oracle.py               (LC-oracle: ceiling, fed gold-attested chunks per question)
│   │   ├── no_context.py              (no-context: floor, closed-book control)
│   │   ├── graphrag/                  (GraphRAG: local + global search; graphrag==3.0.9, stock source)
│   │   └── ie_sql/                    (IE→SQL: schema-gen + Qwen3.6-27B extraction + per-template SQL)
│   ├── infra/                         (Milvus hybrid retriever + Qwen3 embed + reranker + date filter)
│   ├── eval/                          (the scorer used in the paper — strict + tolerant ±7d)
│   ├── prompts/                       (shared system+user prompt scaffold)
│   └── scripts/                       (reproducibility helpers — precache retrieval, verify embedding parity)
│
├── benchmark/                         (QO-Bench)
│   ├── event_definitions.md           (8 corporate-event types, anchored to SEC 8-K items)
│   ├── templates_config.json          (18-template scoring config; paper A/B IDs)
│   └── questions/questions.jsonl.gz   (785 questions: rendered NL text + typed gold)
│
├── extraction/                        (attestation pipeline that built the benchmark's GT)
│   └── attestation_pipeline.py        (3-judge attestation runner; secrets scrubbed)
│
└── docs/
    ├── BENCHMARK.md                   (how the 18 templates + 614 events × 22,984 articles are constructed)
    ├── BASELINES.md                   (per-baseline implementation details — all 5 paradigms + LC-oracle)
    ├── INFRA.md                       (Milvus / vLLM / OpenRouter / GraphRAG environment expectations)
    └── IE_PIPELINE.md                 (the schema-first IE → 3-judge attestation pipeline that produced the GT)
```

## QO-Bench at a glance

- **Corpus**: 22,984 NASDAQ FNSPID news articles (2010–2023)
- **Events**: 614 single-article-attestable corporate events across 8 types
  (M&A {announce, complete, cancel, rumor}, CEO change, CFO change, IPO, stock split)
- **Templates**: 18 query templates (4 Capability A + 14 Capability B)
- **Questions**: 785 (200 Cap A + 585 Cap B), released in `benchmark/questions/questions.jsonl.gz`
- **Ground truth**: deterministic — computed from typed event tuples, no
  LLM-as-judge at scoring time

## What's NOT in this bundle, and why

- **No Milvus host, vLLM endpoint, or OpenRouter API key** — these are
  external services. `docs/INFRA.md` documents what's expected; reviewers
  would need their own infra to fully re-run.
- **No raw S&P Capital IQ KeyDev event rows** — proprietary, license forbids
  redistribution. The benchmark ships only public-identifier event tuples
  (ticker, anchor date, role, counterparty) keyed to FNSPID provenance.
  `docs/IE_PIPELINE.md` documents the upstream.
- **No FNSPID news corpus** (~28 GB) — public dataset, easy to fetch
  independently (see "How to reproduce" below).

The per-paradigm predictions and tolerant-scored `eval_results` referenced in
the paper **are** included under `qobench/baselines/<paradigm>/` for inspection
and re-scoring.

## How to navigate

1. Start with `benchmark/README.md` — what QO-Bench is.
2. Then `docs/BENCHMARK.md` — how the 18 primary templates × 614 events
   produce the questions.
3. Then `docs/BASELINES.md` — what each of the 5 paradigms does + the
   LC-oracle ceiling.
4. To inspect any specific baseline: `qobench/baselines/<name>/` or
   `qobench/baselines/<name>.py`.
5. For the IE pipeline that built the benchmark itself, not the baseline:
   `docs/IE_PIPELINE.md` + `extraction/`.

## How to reproduce

Everything needed to reproduce the paper's numbers is in this folder plus
widely-used public infrastructure (Milvus, vLLM / Hugging Face models, the
public FNSPID corpus). No S&P/Capital IQ access is needed: the gold answers
are released in `benchmark/questions/questions.jsonl.gz` (the `gt` field).

```
# 0. Install (Python 3.10+)
python -m pip install -r requirements.txt

# 1. Corpus. Download the public FNSPID news dataset and keep the 22,984-article
#    subset selected by qobench/infra/v2_url_hash_whitelist.json (URL-hash list).
#    The benchmark questions already carry their rendered natural-language text
#    (the `question` field), produced by qobench/infra/question_renderer.py from
#    (template id, params); rerun that module only if you regenerate questions.

# 2. Answer LLM. Serve Qwen3.6-27B with vLLM and set VLLM_ENDPOINT
#    (e.g. http://localhost:8000/v1), or set OPENROUTER_API_KEY to use a hosted
#    model. The robustness table additionally scores DeepSeek v4-flash / v4-pro
#    via OpenRouter (optional).

# 3. Retrieval index (RAG / ReAct only). Embed the corpus with Qwen3-Embedding-4B
#    (local via sentence-transformers, or DeepInfra) and load into Milvus; set
#    MILVUS_URI / MILVUS_TOKEN. qobench/scripts/precache_retrieval.py caches the
#    per-question top-k. (LC-oracle, no-context, IE->SQL need no Milvus.)

# 4. Run a paradigm -> predictions.jsonl  (each runner takes --benchmark)
python -m qobench.baselines.lc_oracle  --benchmark benchmark/questions/questions.jsonl.gz ...
python -m qobench.baselines.naive_rag  --benchmark benchmark/questions/questions.jsonl.gz ...
#   (also: react_agent, no_context, baselines/graphrag/, baselines/ie_sql/run_ie_sql.py)

# 5. Score against the released gold (tolerant +/-7d, the paper's main metric)
python -m qobench.eval.eval_tolerant \
    --questions   benchmark/questions/questions.jsonl.gz \
    --predictions predictions.jsonl \
    --config      benchmark/templates_config.json \
    --output      results.json
```

GraphRAG's already-run predictions and scores are in
`qobench/baselines/graphrag/{predictions,eval_results}/` as a reference.
LLM decoding is not perfectly deterministic, so reproduced numbers should
match the paper within small run-to-run noise rather than to the digit.

## License

- Code: **MIT** (see `LICENSE`).
- Benchmark questions + IE outputs: **CC-BY-4.0** (see `DATA_LICENSE`).
- Upstream sources (FNSPID, S&P Capital IQ KeyDev): see their original
  licenses; this bundle re-uses derivative outputs only.

## Citation

If you use QO-Bench, please cite:

```bibtex
@misc{zhang2026qobench,
  title         = {QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples},
  author        = {Zhang, Mengao and Yang, Xiang and Liu, Chang and Tan, Tianhui and Huang, Ke-Wei},
  year          = {2026},
  eprint        = {ARXIV_ID},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

(Replace `ARXIV_ID` with the arXiv identifier once the preprint is posted.)
