# Schema-gen prompt (IE→SQL paradigm, LLM-generated schema variant)

For the IE→SQL paradigm of the accompanying paper.

This prompt is used to generate the IE→SQL schema. The schema serves both as (a) the target structure for LLM-based information extraction from news articles, and (b) the database against which natural-language queries are translated to SQL.

**Fairness constraint**: the LLM doing schema-gen sees only `event_definitions.md` (the 8 event types, their authoritative grounding, operational scope, and edge cases) — it does NOT see any benchmark questions, query templates, or task-specific hints. This makes the produced schema a measurement of the LLM's intrinsic schema-design capability, not of engineering investment knowing the questions.

**Escape hatch**: after the LLM produces the schema, the benchmark owner is allowed minimal manual corrections (additions / removals / type changes), each documented. If corrections become non-minimal, the variant falls back to hand-designed schema and the no-leakage claim is downgraded in the paper.

---

## Prompt body (verbatim — paste into model with `event_definitions.md` content appended below)

```
You are designing an event schema to capture all 8 corporate event types
defined in the attached event_definitions.md.

The schema will be:
  (a) Used by an LLM to extract events from individual news articles
      (information extraction target — each record must be derivable from
      a single article).
  (b) Used to answer arbitrary structured queries a domain analyst may pose
      about these events.

You do not have access to specific evaluation questions. Design the schema as
comprehensively as is reasonable — anticipate that an analyst may want to
query any attribute reasonably derivable from a news article. Err toward
including more fields rather than fewer; downstream queries that do not need
a field can simply ignore it.

OUTPUT: a typed schema description with one short inline comment per field.
Briefly note any design trade-offs you weighed.
```

---

## Runner configuration

- **Model**: `openai/gpt-5.5` via OpenRouter
- **System message**: none (prompt body acts as user message)
- **Temperature**: 0 (for reproducibility)
- **Max tokens**: long (8192+ to allow full schema)

## Output location

Save raw model output to `schema_gen_output_<YYYY-MM-DD-HHMM>.md`. The frozen post-escape-hatch schema goes to `schema_v1_frozen.md` after benchmark-owner review.
