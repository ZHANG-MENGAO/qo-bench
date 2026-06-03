# BENCHMARK.md — QO-Bench in detail

## Overview

QO-Bench is a diagnostic benchmark for **query-operator question answering**
(QO-QA): natural-language questions that specify database-style operators
(filter, join, intersect, count, group, order, type-labeled union, empty
return) over typed event tuples latent in a text corpus.

- **Corpus**: 22,984 NASDAQ FNSPID articles (2010–2023).
- **Events**: 614 single-article-attestable events across 8 types.
- **Templates**: 18 (4 Cap A + 14 Cap B).
- **Questions**: 785 (200 Cap A + 585 Cap B).
- **Ground truth**: deterministic — computed from typed event tuples; no
  LLM-as-judge at scoring time.

## File map

| File | Purpose |
|---|---|
| `../benchmark/questions/questions.jsonl.gz` | Canonical 785-question eval set (paper A/B IDs, rendered NL + typed gold) |
| `../benchmark/templates_config.json` | Per-template scoring config (paper A/B IDs) |
| `../benchmark/event_definitions.md` | 8 corporate-event types, anchored to public-record disclosure classes |

## Question record schema

Each line in the JSONL is one question:

```json
{
  "qid": "b587cc6a-3f55-58f5-be7f-1b633f560228",
  "id": "A.1.1",
  "cap": "A",
  "sub_axis": "A.1",
  "params": {"E": "IPO", "W_start": "2013-01-01", "W_end": "2013-12-31"},
  "gt_size": 4,
  "gt": [
    {
      "firm_ticker": "PAGP",
      "event_type": "IPO",
      "anchor_date": "2013-10-15",
      "role": null,
      "counterparty_ticker": null,
      "golden_chunks": [
        {
          "article_url": "...",
          "article_title": "...",
          "article_date": "...",
          "cited_sentence": "...",
          "days_to_event": 0,
          "judge": "3of3",
          "stage": "ipo"
        }
      ]
    }
  ]
}
```

- `id` is the **paper** template ID (the A/B scheme used in the paper's tables);
  released records carry this ID only (no legacy internal IDs).
- `gt[i].golden_chunks` is the per-event provenance — articles that
  3-of-3 attested this event.

## Template taxonomy

### Capability A (filtered retrieval) — 4 primary templates

| ID | Signature | Output | N | Example | Diagnoses |
|---|---|---|---|---|---|
| A.1.1 | (E, W) | List[Event] | 50 | List all CEO changes in Q1 2024. | Type-matched event recall in window. |
| A.1.2 | (E, W) → π_firm | List[Entity] | 50 | Which firms had IPOs in 2018? | Event-to-firm deduplication. |
| A.2.1 | (E, F, W) | List[Event] | 50 | List Microsoft's M&A announcements 2020–2023. | Entity disambiguation under filter. |
| A.3.1 | (E, R, W) | List[Entity/Event] | 50 | Firms that were M&A targets in 2020. | Role-aware retrieval. |

### Capability B (compositional operations) — 14 primary templates

| ID | Signature | Output | N | Diagnoses |
|---|---|---|---|---|
| B.1.1 | (E, F, W, dir) | List[Event] | 50 | Anchor + relative window. |
| B.1.2 | ([E1, E2], W, Δ) | List[EventPair] | 50 | Symmetric temporal join. |
| B.1.3 | ([E_t, E_f], W, Δ) | List[EventPair] | 50 | Directional lag. |
| B.1.4 | (W, Δ) | List[Deal] | 39 | Announce→complete identity. |
| B.1.5 | (F, [E1, E2], W) | Int | 50 | Date arithmetic on self-join. |
| B.2.1 | (E, W, pos) | OrderedList | 50 | Extremal selection. |
| B.2.2 | (F, {E1, ...}, W) | OrderedList | 50 | Multi-type interleaving. |
| B.3.1 | ([E1, E2], W) | List[Entity] | 30 | Cross-type intersection. |
| B.3.2 | (E, R, [W1, W2]) | List[Entity] | 46 | Cross-window same-role. |
| B.3.3 | ([E1, E2], R, W) | List[Entity] | 15 | Cross-type same-role. |
| B.4.1 | (E, W, N) | List[Entity] | 16 | Count threshold. |
| B.4.2 | (E, R, W, N) | List[Entity] | 39 | Role-aware count threshold. |
| B.4.3 | (E, W, b) | List[Event]+b | 50 | Time bucketing. |
| B.4.4 | ({E1, ...}, W) | List[Event]+t | 50 | Type-label union. |

## Scoring

`../qobench/eval/eval.py` and `../qobench/eval/eval_tolerant.py` implement
the strict (±0-day) and tolerant (±7-day) scorers respectively.

The paper's primary metric is **tolerant ±7-day recall on the covered
subset** (gold answers restricted to events the corpus actually attests,
so a system is not penalized for upstream coverage gaps).

## `golden_chunks` field variants

Important schema variability across templates (the scorer handles both):

- **B.1.* templates** use `golden_chunks_{e1, e2, trigger, ...}` (top-level
  fields keyed by entity slot).
- **B.3.1 / B.3.2 / B.3.3 + B.4.1 / B.4.2** use a **nested** structure:
  `qualifying_events: [{event_id, golden_chunks: [...]}]`. The top-level
  `golden_chunks` field is empty for these; the scorer traverses
  `qualifying_events[*].golden_chunks`.

## Reproducing the paper's numbers

1. Generate predictions with any baseline:
   ```
   PYTHONPATH=. python -m qobench.baselines.naive_rag \
     --benchmark benchmark/questions/questions.jsonl.gz \
     --output-dir outputs/
   ```
   (Requires Milvus + vLLM + OpenRouter — see `INFRA.md`.)

2. Score (tolerant ±7d is the paper's primary metric). The scorer is
   stdlib-only and runs standalone:
   ```
   python3 qobench/eval/eval_tolerant.py \
     --questions benchmark/questions/questions.jsonl.gz \
     --predictions outputs/predictions.jsonl \
     --config benchmark/templates_config.json \
     --output outputs/results.json
   ```
