# extraction/ — Attestation pipeline that built QO-Bench's ground truth

This directory contains the **attestation pipeline** that derived the
benchmark's ground-truth event set.

> **Note**: The **IE→SQL baseline** (one of the 5 deployable paradigms
> evaluated in the paper) lives at `../aggqa/baselines/ie_sql/`, not here.
> This directory documents the *upstream* IE pipeline that produces the
> benchmark's typed event tuples (the input to all baselines), not a
> retrieval paradigm.

## Files

- `attestation_pipeline.py` — the runner for the 3-judge attestation that
  produces the operational event set `Ê`. For each candidate (event,
  article) pair, three judges (Gemma-4-31B-IT, Qwen3.6-27B, gpt-oss-120B)
  independently label whether the article attests the event; pairs are
  accepted only on unanimous 3-of-3 confirmation. **Secrets scrubbed**: the
  original script targets an internal MongoDB; set `MONGO_URI` and
  `LLM_URL` env vars to point at your own deployment.

See `../docs/IE_PIPELINE.md` for the full pipeline description (the three
stages, judge models, attestation statistics, and reproduction path).

## What's NOT in this directory and why

See `../docs/IE_PIPELINE.md` for full reproduction guidance. Key omissions:

- Full extraction output (~88 MB) — not bundled; documented for reviewers
  in `../docs/IE_PIPELINE.md`.
- Raw S&P Capital IQ Key Developments event rows — proprietary, not
  redistributed.
- Full FNSPID news corpus (~28 GB) — public dataset, fetch independently.
- Full-corpus attestation outputs (all quarters 2010–2023) — not bundled;
  `attestation_pipeline.py` documents the per-event output format, and the
  paper's `Ê=614` events are derived from the full run.

## How `extraction/` relates to `aggqa/baselines/ie_sql/`

`extraction/` (this directory) is the pipeline that **builds the
benchmark's GT**. It runs the 3-judge attestation funnel:
S&P events × FNSPID articles → ticker+date-window filter → 3-judge
attestation → 614 confirmed events → benchmark.

`aggqa/baselines/ie_sql/` is the **IE→SQL baseline paradigm** that the
benchmark evaluates. It runs an LLM-generated schema, extracts tuples from
articles into it, builds a DuckDB, and answers questions via per-template
SQL skeletons. It is scored against the same gold answers as the other
paradigms.

These are two different uses of LLM-driven IE on the same corpus, with
different purposes.

## License

MIT for `attestation_pipeline.py`. See `../LICENSE` and `../DATA_LICENSE`.
