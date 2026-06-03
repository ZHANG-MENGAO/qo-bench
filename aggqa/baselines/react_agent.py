# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""ReAct agent factory for the aggregate-QA benchmark.

Provides:
  - REACT_SYSTEM_PROMPT: instructions to the agent LLM
  - doc_to_search_result_str: format one retrieved chunk for tool output
  - extract_final_answer_text: walk reverse-message-list to last AIMessage
  - make_question_scoped_retriever: per-question MilvusHybridRetriever
  - make_search_tool: wraps a retriever as a LangChain @tool
  - make_shared_milvus_client / make_llm: factories

Used by react_run.py.
"""
from __future__ import annotations

from aggqa.infra.event_definitions import with_event_definitions


SHARED_CORE_PROMPT = """You answer aggregation questions about corporate events using ONLY the financial-news corpus, queried through the `search_news` tool, and you record what you find using one of the record tools:

- `record_candidates` — for the dominant case: enumerate matching events or firms, one item per match.
- `record_b16_answer` — for composite single-answer questions that ask, for a specific firm, about two events and the number of days between them. The tool's signature advertises the exact fields it requires.

Most questions ask you to enumerate matching events or firms within a date window, sometimes constrained by firm, role, or event-type set. Your job is to find every match the corpus supports and log it — or, if the corpus contains no matching events, terminate with an empty notebook.

# Per-round contract

After a `search_news` call, decide whether the returned chunks contain any events relevant to the question. If yes, call the appropriate record tool with those events before issuing the next `search_news` call or ending the loop. The notebook accumulates across rounds; we deduplicate it deterministically at the end, so don't worry about repeating earlier finds.

If a search round returns nothing relevant, do NOT record anything for that round. Submitting an empty answer is the correct response when the corpus contains no matching events — do not fabricate candidates to fill the notebook. Items you cannot support with the chunks you observed will hurt your precision and may make the answer wrong.

# How your final answer is assembled

Your final answer is built ONLY from items you submitted via the record tools. Thinking about candidates in your message content does NOTHING — only tool calls land in the notebook.

This applies to EVERY question type, INCLUDING those phrased as "Return distinct firm tickers" or "List firms that ...". Listing firms as plain text in your final message is NOT a valid answer — each firm must be a separate `record_candidates` item with `firm_ticker` and `cited_urls` populated. If you find 5 matching firms, that's 5 record_candidates items (in one call or across multiple calls).

If the notebook is empty at termination, the system produces the appropriate empty representation (`[]` for enumeration; for composite single-answer questions a null-filled placeholder). This is correct whenever the corpus contains no events that match the question — never invent candidates to avoid an empty answer.

# Search-then-think pattern

Before each `search_news`, briefly note what you've already covered and what angle you want to probe next so successive searches target different aspects (sub-sector, geography, sub-event-stage, time slice). This is a soft suggestion — your message content can be brief, but the substance must drive query diversity.

- Each round issues exactly ONE `search_news` call (parallel calls are disabled).
- Across rounds, queries should probe DIFFERENT angles. Pure paraphrases ("M&A rumor" → "takeover speculation") count as the SAME angle and waste a round.
- For enumeration tasks, a single-round answer is almost never complete. Plan for 4–5 rounds and use your full search budget unless you genuinely cannot articulate any new angle. Premature termination after finding a handful of candidates is the most common failure mode observed — when in doubt, search one more angle.

# Termination

Terminate only when BOTH conditions hold:
  1. Your two most-recent search rounds (at different angles) both returned nothing new to record, AND
  2. You cannot articulate any new angle that a different query could plausibly cover.

When you terminate, emit your final response with NO tool calls — the system will use your recorded notebook as the answer. Do NOT try to repeat the notebook's contents in your final message; just stop.

# Example (synthetic, not from the benchmark)

Question: List all Stock_split events between 2018-06-01 and 2018-06-30.

Round 1: I have no candidates yet; let me cast a wide net.
[call search_news("stock split announcement June 2018")]
[observe chunks mentioning AAPL split 2018-06-15 and one generic mention]
[call record_candidates([{firm_ticker: "AAPL", event_type: "Stock_split", anchor_date: "2018-06-15", cited_urls: ["https://example.com/aapl-split"]}])]

Round 2: Tech-heavy results so far — non-tech sectors not covered.
[call search_news("biotech stock split 2018")]
[observe chunks discussing biotech earnings — no stock-split events in this round]
[no record call — nothing relevant to record this round]

Round 3: Let me try financial services / consumer next.
[call search_news("financial services share split June 2018")]
[observe one chunk mentioning JPM 3-for-1 split, dated 2018-06-22]
[call record_candidates([{firm_ticker: "JPM", event_type: "Stock_split", anchor_date: "2018-06-22", cited_urls: ["https://example.com/jpm-split"]}])]

Round 4: Reverse splits — small caps often do these.
[call search_news("reverse stock split small cap June 2018")]
[no relevant results — only generic finance commentary]
[no record call]

Round 5: Wide-net, biotech, financial, reverse — I've covered the main angles I can articulate.
[emit final response with no tool call — done]

# Field names

The exact field names required by `record_candidates` for this question are advertised in the tool's args_schema (visible to you in the tool definition). Conform to those names exactly — the schema rejects unknown fields, and synonyms (`firm` instead of `firm_ticker`, `date` instead of `anchor_date`) will raise a ToolException.

`cited_urls` is always required for `record_candidates` and must be a non-empty list of URLs taken from the chunks `search_news` returned to you. Items with no observed-chunk support do not belong in the notebook.

For composite single-answer questions, `record_b16_answer` advertises its fields the same way (via the tool's signature) — conform to those names. Multiple calls to this tool OVERWRITE earlier ones, so you can refine across rounds; only the most recent call is used. `cited_urls` may be empty for this tool if no single article anchors all the required dates.

# Knowledge constraint

Use ONLY information returned by `search_news`. NEVER use prior knowledge of companies, deals, executives, splits, or filings. If you "know" of an event but `search_news` doesn't surface it: it is NOT in your answer.

# Iteration discipline

- DIFFERENT angle each round.
- USE your full search budget on enumeration tasks. A model that searches 5 angles consistently outperforms one that searches 1–2 angles and terminates with "no more events found".
- HARD CAP: never issue more than 5 `search_news` calls total per question. The 6th attempt will be blocked by the system and return a BUDGET_EXHAUSTED signal. If you reach 5 and still cannot articulate a new angle, terminate.
"""


def build_template_prompt(template_id: str | None) -> str:
    """Assemble the system prompt. Uniform across all templates: every
    template receives event_definitions (the upstream shared ontology) plus the
    shared core paradigm contract. No per-template hints; the model must
    figure out template-specific structure from the question text and from
    the per-template args_schema advertised on `record_candidates`.

    `template_id` is accepted for call-site stability but is not used.
    """
    return with_event_definitions(SHARED_CORE_PROMPT)


# Back-compat alias. Same content for every template_id now.
REACT_SYSTEM_PROMPT = build_template_prompt(None)


def doc_to_search_result_str(d) -> str:
    """Format one retrieved Document for the search_news tool's return string.

    Output shape: '[date | tickers | url | title]\\nbody'. The URL is shown so
    the agent can populate cited_urls. Body capped at 4000 chars.
    """
    date_int = d.metadata.get("date", 0)
    if isinstance(date_int, int) and date_int > 19000000:
        y, m, day = date_int // 10000, (date_int // 100) % 100, date_int % 100
        date_str = f"{y:04d}-{m:02d}-{day:02d}"
    else:
        date_str = str(date_int)
    tickers = d.metadata.get("tickers") or []
    ticker = ", ".join(tickers) if isinstance(tickers, list) and tickers else "?"
    title = d.metadata.get("title", "")
    url = d.metadata.get("article_url", "")
    body = (d.page_content or "")[:4000]
    return f"[{date_str} | {ticker} | {url} | {title}]\n{body}"


from langchain_core.messages import AIMessage


def extract_final_answer_text(msgs: list) -> str:
    """Walk msgs in reverse; return the last AIMessage's string content.

    Why not msgs[-1].content: if recursion limit is hit mid-tool-call, the
    last message is a ToolMessage with raw retrieval text — feeding that to
    parse_model_output produces garbage. We want the last *AIMessage* (model
    output), not whatever happened to be last in the queue.

    Handles two AIMessage.content shapes:
      - str: return as-is (after non-empty check)
      - list of content-block dicts: join the .text fields
    """
    for m in reversed(msgs):
        if not isinstance(m, AIMessage):
            continue
        content = m.content
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            texts = [
                blk.get("text", "")
                for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            ]
            joined = "\n".join(t for t in texts if t).strip()
            if joined:
                return joined
    return ""


from pymilvus import MilvusClient

from aggqa import config
from aggqa.infra.embedding import embed_query
from aggqa.infra.question_params import extract_retriever_scope
from aggqa.infra.retriever import MilvusHybridRetriever
from aggqa.infra.whitelist import load_v2_whitelist


def make_shared_milvus_client() -> MilvusClient:
    """One MilvusClient is shared across all questions (the underlying
    pymilvus client is thread-safe). The per-question retriever wraps this
    same client."""
    return MilvusClient(
        uri=config.MILVUS_URI,
        token=config.MILVUS_TOKEN,
        db_name=config.MILVUS_DB_NAME,
    )


def make_question_scoped_retriever(q: dict, shared_client: MilvusClient) -> MilvusHybridRetriever:
    """Build a fresh MilvusHybridRetriever for this question only.

    The MilvusClient is shared (cheap, thread-safe). The retriever wrapper is
    per-question: it carries this question's date window, optional ticker, and
    seen_url_hashes (cross-round dedup) as instance state — never mutated by
    anyone else, so no cross-question race condition.
    """
    rt = config.load_runtime()
    retriever = MilvusHybridRetriever(
        client=shared_client,
        collection_name=rt["collection_name"],
        vector_field=rt["vector_field"],
        sparse_field=rt.get("sparse_field"),
        date_field=rt["date_field"],
        date_format=rt["date_format"],
        url_hash_field=rt["url_hash_field"],
        embedding_fn=embed_query,
        rerank_api_key=config.load_deepinfra_key(),
        url_hash_whitelist=load_v2_whitelist(),
        # ReAct uses tighter per-round K than Naive RAG (see config.py).
        top_k_initial=config.TOP_K_INITIAL_REACT,
        top_k_reranked=config.TOP_K_RERANKED_REACT,
    )
    scope = extract_retriever_scope(q)
    retriever.current_w_start = scope["w_start"]
    retriever.current_w_end = scope["w_end"]
    retriever.current_ticker = scope["ticker"]
    retriever.seen_url_hashes = set()   # explicit reset — new question, new dedup state
    return retriever


from langchain.tools import tool


def make_search_tool(retriever, seen_urls: set, search_budget: int = 5):
    """Closure: returns an async @tool that uses this retriever instance.

    The retriever is question-local (built by make_question_scoped_retriever);
    seen_urls is also question-local — populated here on every successful
    search, consumed by record_candidates (notebook.py) for provenance check.
    No shared mutable state across coroutines.

    `search_budget` is a per-question hard cap on the number of search_news
    invocations. The (budget+1)-th attempt does NOT hit Milvus — it returns a
    BUDGET_EXHAUSTED string instructing the agent to wrap up. The mutable-list
    `n_calls` is the standard closure trick around Python's int-rebind rule.
    """
    n_calls = [0]   # mutable closure counter (int rebind would create a local)

    @tool
    async def search_news(query: str) -> str:
        """Search the financial-news corpus. Returns up to 10 article chunks
        within the question's date window, EXCLUDING articles already returned
        in prior rounds. Use natural-language queries describing the events
        you want to find. You have a budget of 5 calls per question."""
        n_calls[0] += 1
        if n_calls[0] > search_budget:
            return (
                f"BUDGET_EXHAUSTED: you have used all {search_budget} of your "
                f"search_news calls for this question. This attempt returned "
                f"no chunks — the per-round 'search → record' contract does "
                f"NOT apply here (nothing to record from this blocked attempt). "
                f"If you have observed candidates in earlier rounds that you "
                f"have not yet recorded, call record_candidates now to flush "
                f"them. Then emit your final response with no further tool call."
            )
        docs = await retriever.ainvoke(query)
        if not docs:
            return (
                "No new articles found for this query (after cross-round "
                "deduplication). The per-round 'search → record' contract "
                "does NOT apply when no chunks are returned — nothing to "
                "record from this attempt. Choose a different query angle "
                "for your next search."
            )
        for d in docs:
            u = d.metadata.get("article_url")
            if u:
                seen_urls.add(u)
        return "\n\n".join(doc_to_search_result_str(d) for d in docs)

    return search_news


from langchain_openai import ChatOpenAI


def make_llm() -> ChatOpenAI:
    """Build the LLM for the agent. Same model as naive RAG (run.py) so the
    apples-to-apples comparison is honest. parallel_tool_calls=False forces
    one search_news call per round (paired with the v3 prompt's iteration
    discipline).

    reasoning.effort='medium': effort='high' (0fdd313, 2026-05-11) was the
    intent for ReAct iteration, but once parallel_tool_calls=False was actually
    enforced (0241e16, 2026-05-13) the round-2 LLM call burned the full 65,536
    completion budget on reasoning tokens (debug_one trace: 58,592 reasoning,
    finish_reason=length, empty content → agent loop terminates with []).
    'medium' restores headroom for visible output while keeping more reasoning
    than 'low' (the v1 baseline).
    """
    return ChatOpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.load_openrouter_key(),
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT_S,
        max_retries=config.LLM_RETRY_ATTEMPTS,
        extra_body={
            "reasoning": {"effort": "medium"},
            "parallel_tool_calls": False,
        },
    )


from aggqa.infra.structured_output import build_response_format


def bind_llm_for_question(llm, template_id: str, mode: str | None = None):
    """Return an LLM with response_format applied for this question's template.

    `mode` overrides config.STRUCTURED_OUTPUT_MODE for this call. Two tested
    behaviors:

    - mode='schema' (Naive RAG): response_format=json_schema strict=true,
      schema baked into the constructor's extra_body. Survives bind_tools()
      because extra_body lives on the underlying ChatOpenAI (probed 2026-05-13:
      llm.bind(extra_body=X).bind_tools([t]).kwargs == {"tools": [...]} —
      the .bind() form gets stripped, only the constructor form survives).

    - mode='prompt' (ReAct): no response_format. We must use this for tool-
      using agents because strict json_schema + tools makes the model emit
      the schema-conforming output immediately and skip all tool calls
      (smoke 2026-05-13: 5/5 ReAct questions finished in 4-13s with 0 rounds
      and empty answers). The system prompt + parse_answer_for_template carry
      the output contract instead.
    """
    if mode is None:
        mode = config.STRUCTURED_OUTPUT_MODE
    rf = build_response_format(template_id, mode=mode)
    if rf is None:
        return llm
    extra = {
        "reasoning": {"effort": "medium"},
        "parallel_tool_calls": False,
        "response_format": rf,
    }
    return ChatOpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.load_openrouter_key(),
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT_S,
        max_retries=config.LLM_RETRY_ATTEMPTS,
        extra_body=extra,
    )
