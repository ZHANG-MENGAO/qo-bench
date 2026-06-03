# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Prompt templates and helpers for running the v2 QO-Bench benchmark
through a naive-RAG system.

Each benchmark question already contains an output-schema instruction in
its `question` field (per spec §3.5). This module provides:

  - SYSTEM_PROMPT: instructions to the model
  - build_user_prompt(question, retrieved_chunks): assembles the user message
  - parse_model_output(raw_text): robust JSON extraction

Drop-in usage:

    import json
    from prompt_template import SYSTEM_PROMPT, build_user_prompt, parse_model_output

    bench = json.load(open('benchmark_v2_pilot_v2_300q.json'))
    predictions = []

    for q in bench['questions']:
        # 1. Retrieve top-K chunks for q['question'] from your RAG store.
        #    Recommended: pre-filter by date window if your store supports it.
        chunks = retrieve(q, k=30)  # YOUR CODE — see retrieve_hint() below

        # 2. Format prompt
        user_msg = build_user_prompt(q, chunks)

        # 3. Call your LLM
        raw = call_llm(SYSTEM_PROMPT, user_msg)

        # 4. Parse
        answer = parse_model_output(raw)
        predictions.append({'qid': q['qid'], 'answer': answer})

    json.dump({'predictions': predictions}, open('predictions.json', 'w'))
"""

import json
import re
from typing import Optional


# ============================== PROMPTS ==============================

SYSTEM_PROMPT = """You answer questions about corporate events using ONLY the news articles provided in the user message. You must NOT use any prior knowledge of companies, deals, executives, splits, or filings — if the articles don't mention an event, you must not include it.

You output a strict JSON array matching the schema given in the question. No commentary, no markdown fences, no prose, no prefix, no suffix. Just the JSON array starting with [ and ending with ]. If you find no events, output []."""


PUBLIC_EVENT_ONTOLOGY = """# Public event-type ontology (use these exact strings, case-sensitive, in the `event_type` field)

  M&A_rumor, M&A_announce, M&A_complete, M&A_cancel
  CEO_change_announce, CEO_change_effective
  Stock_split, IPO_pricing
  Chapter_11_filing, DIP_financing, restructuring, emergence, liquidation
"""


USER_PROMPT_TMPL = """{ontology}
# Retrieved articles (top-{k}, ordered by retrieval relevance)

{retrieved_block}

# Question

{question_text}
"""


# ============================== BUILDERS ==============================

def format_chunk(idx: int, chunk: dict, max_body_chars: int = 1500) -> str:
    """Format one retrieved chunk for the prompt.

    Expected chunk fields:
      date           : str (article date, ISO preferred)
      ticker         : str (associated ticker, optional)
      title          : str
      body           : str (article text or chunk text)

    Override this if your chunk schema is different.
    """
    body = (chunk.get('body') or chunk.get('article_text') or chunk.get('text') or '').strip()
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + ' …[truncated]'
    title = chunk.get('title') or chunk.get('article_title') or '(untitled)'
    dt = chunk.get('date') or chunk.get('article_date') or ''
    if dt:
        dt = str(dt)[:10]
    tk = chunk.get('ticker') or chunk.get('symbol') or '?'
    return f"[{idx}] {dt} | {tk} | {title}\n{body}"


def build_user_prompt(question: dict, retrieved_chunks: list,
                      k: Optional[int] = None,
                      max_body_chars: int = 1500) -> str:
    """Assemble the user message for one question + retrieved context.

    question: a dict from benchmark['questions'] — uses `question` field,
        which already contains the output schema.
    retrieved_chunks: list of chunk dicts (your RAG retrieval output).
    """
    if k is None:
        k = len(retrieved_chunks)
    retrieved_block = "\n\n".join(
        format_chunk(i, c, max_body_chars=max_body_chars)
        for i, c in enumerate(retrieved_chunks[:k], start=1)
    )
    return USER_PROMPT_TMPL.format(
        ontology=PUBLIC_EVENT_ONTOLOGY,
        k=k,
        retrieved_block=retrieved_block,
        question_text=question['nl_question'],
    )


# ============================== PARSING ==============================

def parse_model_output(raw_text: str) -> list:
    """Robust JSON-array extraction. Handles:
       - markdown ```json ... ``` fences
       - leading/trailing prose
       - merely-quoted JSON

    Returns list of tuple-dicts, possibly empty. Eval script does its own
    normalization, so don't worry about case / date format here.
    """
    if raw_text is None:
        return []
    s = str(raw_text).strip()
    # Strip markdown fences
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    # Slice from first '[' to last ']'
    fb = s.find('[')
    lb = s.rfind(']')
    if fb < 0 or lb <= fb:
        return []
    s = s[fb:lb + 1]
    try:
        parsed = json.loads(s)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [t for t in parsed if isinstance(t, dict)]


# ============================== RETRIEVAL HINTS ==============================

def retrieve_hint():
    """Reference notes for setting up retrieval — not executed.

    The benchmark questions reference a time window (W_start, W_end) and an
    event type. For a fair "naive RAG" baseline:

    1. **Pre-filter by date window** in your retrieval (anchor_date >=
       W_start - 30d AND <= W_end + 30d, say). Without this, naive RAG will
       pull articles from outside the window and the recall floor gets
       artificially low.

    2. **Top-K = 20 to 50.** With Milvus + Qwen3 dense + BM25 sparse hybrid,
       30 is a good starting point.

    3. **Use group_by_field='url_hash'** in Milvus to dedupe to article level
       (chunks of the same article share url_hash).

    4. Build the retrieval query by concatenating the question text with the
       event_type tokens and any firm-name hints from the question. Don't
       do query rewriting / decomposition / multi-hop — that's not naive
       RAG anymore.

    5. For T26 (zero-case) questions, retrieval will still return *something*
       (closest matches in the corpus). The model should look at the
       articles and decide they don't actually describe the queried event,
       then return []. This tests calibrated refusal.
    """
    pass
