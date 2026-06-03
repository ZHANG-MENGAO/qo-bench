# GraphRAG Paradigm — QO-Bench

GraphRAG (microsoft/graphrag) implementation of the QO-Bench paradigm
comparison. Both **local** and **global** search modes are evaluated on the
cap=50 benchmark (785 questions, 18 templates).

## TL;DR results

Headline numbers (paper primary metric: tolerant ±7d, micro-averaged recall
on the covered subset):

|             | Cap A (n=200) | Cap B (n=585) | Overall (n=785) |
|-------------|---------------|---------------|-----------------|
| GR-local    | 2.6%          | 0.3%          | 0.9%            |
| GR-global   | 7.1%          | 2.7%          | 3.8%            |

## Repository layout

```
graphrag/
├── README.md                  ← you are here
├── run_query.py               ← orchestration: claim-coordinated cap50 batch runner
├── inject_event_definitions.py← injects shared task ontology into prompts
├── settings.yaml              ← graphrag config (non-stock deviations annotated)
├── env.example                ← copy to .env; vLLM API endpoints (any non-empty key works)
├── prompts/
│   ├── local_search_system_prompt.txt        ← stock + (a) task ontology + (b) URL-cite patch
│   ├── global_search_map_system_prompt.txt   ← stock + (a) task ontology
│   ├── global_search_reduce_system_prompt.txt← stock + (a) task ontology
│   ├── drift_search_system_prompt.txt        ← stock + (a) + (b)
│   └── drift_reduce_prompt.txt               ← stock + (b)
├── predictions/
│   ├── predictions_local.jsonl          ← 785 GR-local predictions
│   └── predictions_global_final.jsonl   ← 785 GR-global predictions (2 stubborn qids → [])
└── eval_results/
    ├── eval_local_tolerant_v2.json      ← scorer output for local
    └── eval_global_final_tolerant_v2.json
```

## Reproducibility

### Stack

- `graphrag==3.0.9` (microsoft/graphrag, **source code unmodified**)
- vLLM 0.19.1 serving:
  - chat: Qwen3.6-27B (`--reasoning-parser qwen3`)
  - embed: Qwen3-Embedding-4B
- Index uses the same 22,984-article public corpus as RAG / ReAct / IE→SQL baselines (substrate not bundled here; see benchmark distribution)

### Index settings (non-stock deviations)

See `settings.yaml`. Key changes vs stock graphrag 3.0.9:
- `chunking.prepend_metadata = [article_url, article_date, article_primary_ticker, article_primary_company_name]`
- `extract_graph.max_gleanings = 2`
- `summarize_descriptions.max_length = 1000`
- `extract_claims.enabled = true`
- `prune_graph.min_node_freq = 1`, `min_edge_weight_pct = 0`
- `concurrent_requests = 128` (workflow-level throughput)

### Query settings

graphrag 3.0.9 **stock defaults** for local/global/drift (no overrides):
`top_k_mapped_entities=10`, `community_level=2`, etc.

### Prompt modifications (disclosed)

5 of 12 user-facing prompts modified:
- **(a) Event ontology prepend** — paradigm-agnostic shared task description (the same ontology fed to LC / RAG / ReAct / IE→SQL baselines). Files: `local_search_*.txt`, `global_search_*.txt` (map + reduce), `drift_search_*.txt`.
- **(b) URL-aware citation patch** — instructs LLM to extract `article_url:` lines from source chunks for the `cited_urls` output field, instead of graphrag's stock `Sources (N)` ID format which cannot be evaluated for article-level provenance. Files: `local_search_*.txt`, `drift_search_*.txt`, `drift_reduce_prompt.txt`.

Stock prompts (7 of 12) are not bundled — see microsoft/graphrag upstream repo for verbatim originals.

### Running

1. Bring up vLLM servers (chat on port 8000, embed on 8001). Set `.env` accordingly.
2. Build the index from the corpus (`graphrag index --root .` after placing `articles.csv` under `input/`). This step is expensive (~16h with 4 GPUs and `extract_claims=true`).
3. Run queries via `run_query.py`, which calls `graphrag query` as subprocess per question. The orchestrator supports filesystem-claim coordination for multi-worker parallelism.
4. Score with the bundled stdlib-only scorer:
   ```
   python3 ../../eval/eval_tolerant.py \
     --questions ../../../benchmark/questions/questions.jsonl.gz \
     --predictions predictions/predictions_global_final.jsonl \
     --config ../../../benchmark/templates_config.json \
     --output eval_results/eval_global_final_tolerant_v2.json
   ```

## Caveats

- Both methods have provenance recall ≈ 0% (architectural limit for global; near-chance for local).
- 2 stubborn global qids (B.1.3, B.2.1) never reached by any worker within budget; treated as `[]` (= 0 recall) in final numbers.
- Article-level **retrieval** recall (not answer recall) not exposed by graphrag 3.0.9 CLI; computed cited-based proxy at 0.39% local / ~0% global (lower bound only).
