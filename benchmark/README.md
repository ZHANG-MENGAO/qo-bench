# benchmark/ — QO-Bench

The **QO-Bench** benchmark: a diagnostic suite for *query-operator question
answering* (QO-QA) over typed corporate-event tuples latent in financial
news.

## Files

- `event_definitions.md` — the **8 corporate-event types** covered, each
  anchored to a public-record disclosure class (SEC 8-K items, the
  Securities Act, NYSE/NASDAQ listing rules). Supplied verbatim to every
  baseline paradigm.
- `templates_config.json` — per-template scoring config (identity keys,
  scoring type, golden-chunks field variant), keyed by the paper's A/B
  template IDs.
- `questions/questions.jsonl.gz` — **the canonical evaluation set**: 785
  questions (200 Cap A + 585 Cap B), cap=50 stratified sample. Each record
  carries its rendered natural-language `nl_question`, typed `gt` (gold
  denotation), `params`, and paper template `id`.

## How questions are constructed

Each question record is a JSONL row with these fields (see
`../docs/BENCHMARK.md` for full detail):

| Field | Meaning |
|---|---|
| `qid` | Question UUID |
| `id` | Paper template ID (A/B scheme, e.g., `A.1.1`, `B.1.4`) |
| `cap` | `"A"` (filtered retrieval) or `"B"` (compositional ops) |
| `sub_axis` | Sub-axis within capability (e.g., `A.1`, `B.4`) |
| `params` | Typed parameters: `E` (event type), `W_start`/`W_end` (date window), `F` (firm), `R` (role), etc. |
| `gt_size` | Number of items in ground truth |
| `gt` | Ground-truth list with per-item `golden_chunks` provenance |

Each question is generated deterministically from
`(template_id, event_set, params)`:

1. Select events from the operational set `Ê` (614 single-article-attestable
   events derived by the IE pipeline — see `../docs/IE_PIPELINE.md`).
2. Apply the template's predicate (date window, optional firm filter, etc.).
3. Render the natural-language question.
4. Compute the gold answer denotation as the template's operator over the
   selected tuples.
5. Persist gold chunks (the news articles that 3-of-3 attested each
   contributing event) as evidence metadata.

## Statistics (from paper)

- **22,984** NASDAQ FNSPID articles (2010–2023)
- **614** single-article-attestable corporate events across 8 types
- **18** query templates (4 Capability A + 14 Capability B)
- **785** questions in the released cap=50 file (200 Cap A + 585 Cap B)

## Two capability classes

- **Capability A (filtered retrieval)**: A.1.1, A.1.2, A.2.1, A.3.1 — single
  aggregator operations: list, count, dedup, role-aware filter.
- **Capability B (compositional operations)**: B.1.*, B.2.*, B.3.*, B.4.* —
  add an operator over filtered events: temporal join, intersection,
  ordering, count threshold, grouping, type-labeled union.

See `../docs/BENCHMARK.md` for the full template catalog (signatures,
sample sizes, examples, diagnostic targets).

## License

CC-BY-4.0. See `../DATA_LICENSE` and the upstream attribution requirements
documented there.
