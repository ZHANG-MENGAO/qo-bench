# Responsible NLP Research Checklist — QO-Bench (Zhang et al., 2026)

This checklist follows the ACL Rolling Review Responsible NLP Research
Checklist (https://aclrollingreview.org/responsibleNLPresearch/). Each item
is answered explicitly; pointers reference the paper PDF and this
supplementary bundle.

Paper title: *QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over
Typed Event Tuples*.

---

## A. Every submission

### A1. Did you describe the limitations of your work?
**Yes.** See the paper's "Limitations" section. Key items: corporate-events
ground truth may not transfer to domains with weaker public records or noisier
entity resolution; the benchmark uses template-generated questions that
underrepresent linguistic diversity; tuple ground truth still relies on
extraction/judging that may contain residual errors (mitigated via public
identifiers, unanimous 3-judge attestation, and human validation); paradigm
implementations are one of many possible, so results are matched-condition
architecture diagnostics, not model-invariant claims.

### A2. Did you discuss any potential risks of your work?
**Yes.** See the paper's "Ethics and Responsible Benchmarking" section. The
benchmark is built from public corporate news and public corporate-event
records; no non-public personal data. Coverage skews toward large, publicly
listed firms, English-language markets, and media-salient events. The
benchmark is intended for retrieval-architecture evaluation, not for
investment, legal, compliance, or employment decisions.

---

## B. Scientific artifacts (most relevant — we ship a benchmark, baselines, IE pipeline)

### B1. Did you cite the creators of the artifacts you used?
**Yes.** FNSPID news corpus (`\cite{dong2024fnspid}`); S&P Capital IQ Key
Developments (cited as the proprietary upstream); Microsoft graphrag
(`\cite{edge2024graphrag}`); LangChain, vLLM, pymilvus,
sentence-transformers, transformers, DuckDB cited as software dependencies.

### B2. Did you discuss the license or terms of use of any artifacts?
**Yes.** See `DATA_LICENSE` at bundle root and the paper's
Appendix~\ref{app:disclaimer}.
- **FNSPID**: public; we use the NASDAQ subset.
- **S&P Capital IQ Key Developments**: **proprietary**, not redistributed
  in any form. Only public-identifier event tuples (ticker, anchor date,
  role, counterparty, public-record disclosure class) plus FNSPID
  provenance article IDs are released. Reconstructing the original vendor
  records requires a separate S&P Capital IQ license.
- **Our derivative outputs** (questions, gold answers, baseline
  predictions): CC-BY-4.0.
- **Code**: MIT.

### B3. Is the use consistent with their intended use?
**Yes.** FNSPID was released for academic NLP research on financial news.
S&P Capital IQ Key Developments is licensed for academic research at our
institution and is used only to seed event tuples; no S&P content is
republished. The Microsoft graphrag library is open-source and intended for
exactly this kind of benchmark.

### B4. Have you taken steps to verify that no PII or offensive content
is present?
**Partial — documented.** News articles can contain public-figure names
(executives, board members, CEOs/CFOs); these are public figures and the
articles are already in the public domain. No automated PII filtering
applied. No automated offensive-content filtering applied; the
financial-news domain is low risk for offensive content. The benchmark
questions are auto-generated from structured templates and contain no
free-text crowdsourced content.

### B5. Did you provide documentation of the artifacts?
**Yes.**
- `benchmark/README.md` documents the benchmark structure.
- `docs/BENCHMARK.md` documents domain (U.S.-listed equity-issuer
  corporate events), language (English), time span (2010–2023).
- `docs/BASELINES.md` documents each of the 5 deployable baselines plus the
  LC-oracle ceiling.
- `docs/IE_PIPELINE.md` documents the schema-first IE → 3-judge attestation
  flow that produces the benchmark's ground truth.
- `docs/INFRA.md` documents external-service expectations (Milvus, vLLM,
  OpenRouter, GraphRAG environment).

### B6. Did you report relevant statistics (number of examples, train/test
splits, etc.)?
**Yes.**
- **Templates**: 18 (4 Capability A + 14 Capability B).
- **Events**: 614 single-article-attestable events, drawn from a candidate
  pool of 1,376 S&P events (originally selected from 16,414 events of the
  eight types across 2010–2023). Of 25,888 candidate (event, article)
  pairs, 1,591 (6.1%) were attested 3-of-3.
- **Articles**: 22,984 distinct FNSPID articles (one article averages 1.13
  events).
- **Event types**: 8 types — M&A {announce, complete, cancel, rumor}, CEO
  change, CFO change, IPO, stock split.
- **Questions**: 785 (200 Cap A + 585 Cap B), cap=50 stratified sample.
- **Domain**: U.S.-listed equity-issuer corporate events.
- **Language**: English.
- **No standard train/test split**: this is an eval-only benchmark.

---

## C. Computational experiments

### C1. Did you report the number of parameters in your models and the
compute budget?
**Yes.** See `docs/BASELINES.md`, `docs/INFRA.md`, and the paper appendix:
- **Answer model**: Qwen3.6-27B (open weights) for all 5 deployable
  paradigms and the LC-oracle ceiling.
- **vLLM serving**: 0.17.1 for the 4 LLM/RAG baselines (2-GPU tensor
  parallel, bfloat16, FlashAttention-3); 0.19.1 for GraphRAG.
- **Retrieval**: Qwen3-Embedding-4B (4B params, 2560-dim dense) +
  Qwen3-Reranker-4B (4B params) for RAG and ReAct RAG.
- **IE→SQL Stage 1 schema generator**: GPT-5.5 (T=0).
- **GraphRAG conda env**: ~12 GB total (graphrag 1.2 GB + vllm_server 11 GB);
  GraphRAG indexing ~16h on 4 GPUs with `extract_claims=true`.
- **ReAct**: OpenRouter for Qwen3.6-27B (provider routing not pinned).
- Cluster job scripts and walltimes are deployment-specific and not
  bundled; `docs/INFRA.md` documents the external-service requirements.

### C2. Did you discuss hyperparameter search and tuning?
**Yes.** No hyperparameter tuning was performed for this position paper.
All baselines use greedy decoding (`temperature=0`) with a single seed.
Decoding details per baseline are in `docs/BASELINES.md`. GraphRAG uses
stock graphrag-3.0.9 defaults for local/global/drift search modes (no
overrides). RAG uses top-30 reranked chunks. ReAct issues up to 5 retrieval
calls per question.

### C3. Did you report descriptive statistics, including variance?
**See paper.** Primary metric: micro-averaged recall under ±7-day date
tolerance, on the covered subset (gold answers restricted to events
attested in the corpus). Strict ±0-day and provenance-aware variants are in
the paper's appendix. Greedy single-seed decoding means there is no
run-to-run variance to report.

### C4. Did you report the package versions you used?
**Yes.** Pinned in `requirements.txt`. Key versions: vLLM 0.17.1 (LLM/RAG)
and 0.19.1 (GraphRAG), langchain >=1.0 <2, pymilvus >=2.5 <2.6,
microsoft graphrag 3.0.9, DuckDB >=0.10.

---

## D. Human annotators

### D1–D5: Did you use crowdworkers / human annotators?
**Yes — for attestation precision validation only.** Three human annotators
(one primary expert + two secondary reviewers, all academic collaborators
on the project, paid via project research budget — not crowdsourced)
re-examined a stratified sample of N_val=221 accepted (event, article)
pairs to validate the 3-of-3 LLM-judge consensus. Each pair received at
least two independent labels. Result: 3-of-3 LLM consensus achieves 94.3%
precision against expert labels (Cohen's κ=0.538 inter-annotator agreement).
See paper §3.2 (Validating the Judge Consensus). The benchmark's gold
answers themselves are computed deterministically from normalized event
tuples — no human in the scoring loop.

---

## E. AI assistants

### E1. Did you use AI assistants in writing or coding?
**Yes.** Claude Code (Anthropic Claude Opus 4.7) was used as a coding
assistant for substantial portions of the supplementary bundle, the 4
LLM/RAG baselines, and the IE→SQL baseline implementation. Every code
commit and prose passage was reviewed and approved by a human author before
being merged.
