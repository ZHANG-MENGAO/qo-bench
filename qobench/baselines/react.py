# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""ReAct baseline runner: same retrieval backend and predictions.json schema
as run.py, but uses an agentic ReAct loop (langchain.agents.create_agent)
instead of the single-shot retrieve-then-generate pattern.

Usage:
  python -m qobench.react_run --smoke 5
  python -m qobench.react_run --output-dir outputs_react_v1
  python -m qobench.react_run --benchmark path/to/benchmark.json

Output format matches run.py exactly so the existing eval.py works untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback

from langchain.agents import create_agent
from langchain_core.callbacks.base import AsyncCallbackHandler
from langgraph.errors import GraphRecursionError

from qobench import config
from qobench.infra.benchmark_loader import load_benchmark
from qobench.infra.question_renderer import render_question
from qobench.baselines.react_agent import (
    build_template_prompt,
    bind_llm_for_question,
    extract_final_answer_text,
    make_llm,
    make_question_scoped_retriever,
    make_search_tool,
    make_shared_milvus_client,
)
from qobench.baselines.notebook import (
    build_candidate_schema,
    finalize_notebook,
    load_template_cfg,
    make_record_candidates_tool,
    make_record_b16_answer_tool,
)
from qobench.baselines.naive_rag import format_jsonl_record


DEFAULT_CONCURRENCY = 10
# Empirically each round ≈ 2 supersteps (model node + tool node), not the 4
# anticipated by spec §3.3. 21q smoke 2026-05-15 showed real runs taking 7-9
# rounds (B.1.1, B.1.5, B.2.2, B.3.3) and 2/21 hitting the prior 18-cap with
# productive notebooks (A.3.combined had 14 candidates recorded when cut off,
# B.4.1 had 11). 30 supersteps ≈ 15 rounds — covers all observed runs plus
# headroom for LangGraph book-keeping. Keep bounded so a runaway loop still
# terminates within a question budget.
RECURSION_LIMIT = 30
# Per spec §2: tightened from 1500. 4 rounds × ~145s ≈ 580s baseline; 1200
# leaves ~2× buffer. Hard guarantee — criterion §6.10.
PER_QUESTION_TIMEOUT_S = 1200

# OpenRouter occasionally returns truncated/malformed JSON in the response
# body (observed 2026-05-14 on B.1.2; raised inside openai SDK's
# raw_response.parse() as json.JSONDecodeError). The OpenAI client's
# max_retries only retries API/5xx/timeout/rate-limit — parse failures
# bypass it. We retry once at our layer; finalize_notebook dedups (see
# notebook.py:77-98), so the agent re-running its prior tool calls is safe.
TRANSIENT_RETRY_MAX_ATTEMPTS = 2
TRANSIENT_RETRY_BACKOFF_S = 2.0


async def _invoke_with_json_retry(agent, state, *, config):
    """Call agent.ainvoke; retry once on json.JSONDecodeError.

    Why a custom retry: see TRANSIENT_RETRY_* module-level comment.
    Why safe to retry: notebook is closure-captured by record_* tools and
    persists across attempts; finalize_notebook dedups by identity_keys.
    """
    import json as _json
    last_exc: Exception | None = None
    for attempt in range(1, TRANSIENT_RETRY_MAX_ATTEMPTS + 1):
        try:
            return await agent.ainvoke(state, config=config)
        except _json.JSONDecodeError as e:
            last_exc = e
            if attempt >= TRANSIENT_RETRY_MAX_ATTEMPTS:
                raise
            if TRANSIENT_RETRY_BACKOFF_S > 0:
                await asyncio.sleep(TRANSIENT_RETRY_BACKOFF_S * attempt)
    raise last_exc  # unreachable


def _stringify_content(c) -> str:
    """Coerce AIMessage.content (str | list[block]) to a plain string."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for blk in c:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
            elif isinstance(blk, dict):
                parts.append(f"[{blk.get('type','?')}]")
        return "\n".join(parts)
    return str(c)


def _extract_flat_transcript(msgs) -> list[dict]:
    """One entry per message: AIMessage with reasoning + tool_calls,
    ToolMessage with name + full content, HumanMessage with question text.
    Preserves order so a reader can replay the agent turn-by-turn."""
    from langchain_core.messages import (
        AIMessage, ToolMessage, HumanMessage, SystemMessage,
    )
    out: list[dict] = []
    for m in msgs:
        kind = type(m).__name__
        entry: dict = {"kind": kind}
        if isinstance(m, AIMessage):
            entry["content"] = _stringify_content(m.content)
            entry["tool_calls"] = [
                {
                    "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?"),
                    "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}),
                }
                for tc in (m.tool_calls or [])
            ]
        elif isinstance(m, ToolMessage):
            entry["tool_name"] = getattr(m, "name", None)
            c = m.content
            entry["content"] = c if isinstance(c, str) else str(c)
        elif isinstance(m, (HumanMessage, SystemMessage)):
            entry["content"] = _stringify_content(m.content)
        out.append(entry)
    return out


class _TranscriptRecorder(AsyncCallbackHandler):
    """Failure-path safety net: captures every on_tool_start (with parsed
    inputs) and on_tool_end (with output content). Used to reconstruct the
    agent's query/chunk history when result["messages"] is unavailable
    (timeout, recursion_limit, exception inside ainvoke).

    Memory note `feedback_observability_must_survive_failures`: failure paths
    matter most — this recorder is the only way to see what an agent did
    before crashing."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self._pending: dict = {}  # run_id -> {name, inputs}

    async def on_tool_start(self, serialized, input_str, *, run_id=None,
                            inputs=None, **kw) -> None:
        name = (serialized or {}).get("name", "?")
        self._pending[run_id] = {
            "name": name,
            "inputs": inputs if inputs is not None else input_str,
        }

    async def on_tool_end(self, output, *, run_id=None, **kw) -> None:
        start = self._pending.pop(run_id, {"name": "?", "inputs": None})
        content = getattr(output, "content", None)
        if content is None:
            content = str(output)
        self.events.append({
            "name": start["name"],
            "inputs": start["inputs"],
            "output": content,
        })

    async def on_tool_error(self, error, *, run_id=None, **kw) -> None:
        start = self._pending.pop(run_id, {"name": "?", "inputs": None})
        self.events.append({
            "name": start["name"],
            "inputs": start["inputs"],
            "error": repr(error),
        })


def _dump_transcript(qid: str, q: dict, *,
                     output_dir, system_prompt: str,
                     msgs=None, recorder=None,
                     log_record: dict | None = None,
                     final_answer=None) -> None:
    """Write transcripts/{qid}.json. Prefer msgs (success path, complete with
    AIMessage reasoning); fall back to recorder.events (failure path, tool I/O
    only — no model reasoning text)."""
    out_dir = output_dir / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript: dict = {
        "qid": qid,
        "template_id": q.get("id"),
        "question_text": render_question(q),
        "params": q.get("params"),
        "gt_size": q.get("gt_size"),
        "system_prompt": system_prompt,
        "final_answer": final_answer,
    }
    if log_record:
        transcript["stats"] = {
            k: log_record.get(k) for k in (
                "n_search_calls", "n_record_calls", "n_tool_rounds",
                "n_tool_calls", "search_budget_hit", "parsed_n_tuples",
                "notebook_size",
            )
        }
        if "error" in log_record:
            transcript["error"] = log_record["error"]
        if "traceback" in log_record:
            transcript["traceback"] = log_record["traceback"]

    if msgs is not None:
        transcript["source"] = "messages"
        transcript["messages"] = _extract_flat_transcript(msgs)
    elif recorder is not None:
        transcript["source"] = "callback_partial"
        transcript["tool_events"] = recorder.events
    else:
        transcript["source"] = "empty"

    path = out_dir / f"{qid}.json"
    path.write_text(json.dumps(transcript, ensure_ascii=False, indent=1))


async def process_question(q, llm, shared_client, agent_sem):
    """Run the ReAct agent on one question; return (qid, prediction, log).

    Per-question state (no cross-coroutine sharing):
      - retriever: built by make_question_scoped_retriever
      - notebook: list[dict] populated by record_candidates / record_b16_answer
      - seen_urls: set[str] populated by search_news; consumed by record_* for
        the provenance check.

    Tool binding is UNIFORM across templates: every question gets all three
    tools (search_news + record_candidates + record_b16_answer). The model
    picks the right record tool based on the question. This makes the baseline
    fair: hallucination resistance on empty_check templates is now actually
    measurable (model can record candidates → can hallucinate → can score 0),
    not structurally guaranteed via tool absence.

    Answer comes from finalize_notebook(notebook, template_cfg) — deterministic
    Python dispatch, no parse of the final AIMessage. The agent loop's natural
    termination (final AIMessage with no tool_calls) ends the loop; we don't
    use the final AIMessage's text content for anything.

    Catches GraphRecursionError / asyncio.TimeoutError / other Exception:
    finalize_notebook still runs on whatever's in the notebook (better than []
    when we had partial progress).
    """
    qid = q["qid"]
    template_id = q["id"]
    nl_question = render_question(q)
    template_cfg = load_template_cfg(template_id)
    log_record = {"qid": qid, "template_id": template_id}

    # Per-question state
    notebook: list[dict] = []
    seen_urls: set[str] = set()
    counter = _ToolCounter(search_budget=config.REACT_SEARCH_BUDGET)
    recorder = _TranscriptRecorder()
    sys_prompt = build_template_prompt(template_id)
    msgs_for_dump = None  # populated on success path

    try:
        retriever = make_question_scoped_retriever(q, shared_client)
        search_tool = make_search_tool(
            retriever, seen_urls, search_budget=config.REACT_SEARCH_BUDGET
        )

        # Uniform tool binding: every template gets all three tools. The
        # model picks the right record tool based on the question.
        # build_candidate_schema falls back to a generic event schema for
        # empty_check templates (which have no items.properties in
        # templates_config) — see notebook.py.
        schema_cls = build_candidate_schema(template_cfg)
        debug_dump_path = None
        if os.environ.get("AGGREGATE_QA_NOTEBOOK_DUMP") == "1":
            debug_dump_path = config.OUTPUTS_DIR / f"notebook_raw_{qid}.jsonl"
        tools = [
            search_tool,
            make_record_candidates_tool(
                notebook, seen_urls, schema_cls, debug_dump_path=debug_dump_path
            ),
            make_record_b16_answer_tool(notebook, seen_urls),
        ]

        # mode='prompt': strict json_schema + tools collapses the agent loop
        # (smoke 2026-05-13). Naive RAG still uses schema mode.
        bound_llm = bind_llm_for_question(llm, template_id, mode="prompt")
        agent = create_agent(
            model=bound_llm,
            tools=tools,
            system_prompt=sys_prompt,
        )
        async with agent_sem:
            result = await asyncio.wait_for(
                _invoke_with_json_retry(
                    agent,
                    {"messages": [{"role": "user", "content": nl_question}]},
                    config={
                        "recursion_limit": RECURSION_LIMIT,
                        "callbacks": [counter, recorder],
                    },
                ),
                timeout=PER_QUESTION_TIMEOUT_S,
            )

        msgs = result.get("messages", [])
        msgs_for_dump = msgs
        n_search_calls, n_record_calls = _count_tool_calls(msgs)

        # Retry mechanism removed (2026-05-18 prompt refactor): the previous
        # "you searched but didn't record — record now" retry presumed the
        # only legitimate reason for an empty notebook was that the model
        # forgot. The new SHARED_CORE_PROMPT explicitly licenses "empty
        # notebook" as the correct response when no observed chunks support
        # any answer — so the retry's heuristic (search>0 && notebook==0
        # = probably forgot) is no longer reliable. We trust the prompt and
        # the model.
        log_record.update({
            "n_messages": len(msgs),
            "msg_types": [type(m).__name__ for m in msgs],
            # Tool-call counts come from the callback counter, which is the
            # only source-of-truth that survives the failure paths below.
            "notebook_size": len(notebook),
            "seen_urls_size": len(seen_urls),
            "retried": False,  # kept for log-schema stability; always False now
            "final_text": extract_final_answer_text(msgs)[:5000],
        })
        log_record.update(counter.snapshot())
        answer = finalize_notebook(notebook, template_cfg)
        log_record["parsed_n_tuples"] = len(answer) if isinstance(answer, list) else 1
        _dump_transcript(qid, q, output_dir=config.OUTPUTS_DIR,
                         system_prompt=sys_prompt, msgs=msgs_for_dump,
                         recorder=recorder, log_record=log_record,
                         final_answer=answer)
        return qid, {"qid": qid, "answer": answer}, log_record
    except asyncio.TimeoutError:
        log_record["error"] = f"timeout: {PER_QUESTION_TIMEOUT_S}s"
    except GraphRecursionError:
        log_record["error"] = f"recursion_limit_hit: {RECURSION_LIMIT}"
    except Exception as e:
        # Preserve full traceback so we can diagnose without re-running.
        # See memory: feedback_preserve_full_traceback.
        log_record["error"] = repr(e)
        log_record["traceback"] = traceback.format_exc()

    # Failure path: finalize whatever made it into the notebook.
    log_record.setdefault("retried", False)  # legacy field; always False
    log_record.update(counter.snapshot())
    answer = finalize_notebook(notebook, template_cfg)
    log_record["notebook_size"] = len(notebook)
    log_record["parsed_n_tuples"] = len(answer) if isinstance(answer, list) else 1
    _dump_transcript(qid, q, output_dir=config.OUTPUTS_DIR,
                     system_prompt=sys_prompt, msgs=msgs_for_dump,
                     recorder=recorder, log_record=log_record,
                     final_answer=answer)
    return qid, {"qid": qid, "answer": answer}, log_record


def _count_tool_calls(msgs) -> tuple[int, int]:
    """Return (n_search_calls, n_record_calls) — names matched literally."""
    n_search = 0
    n_record = 0
    for m in msgs:
        tcs = getattr(m, "tool_calls", None) or []
        for tc in tcs:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name == "search_news":
                n_search += 1
            elif name in ("record_candidates", "record_b16_answer"):
                n_record += 1
    return n_search, n_record


class _ToolCounter(AsyncCallbackHandler):
    """Real-time tool-call counter. Survives agent.ainvoke failures (timeout,
    recursion_limit, unexpected exception) because the counters live on the
    handler instance owned by process_question, not in the unreturned
    `result["messages"]`. Without this, failure paths log `rounds=?` — exactly
    the cases that matter most for analyzing agent behavior.

    LangGraph callback-propagation caveat (verified 2026-05-15 by direct
    trace): callbacks supplied via `config["callbacks"]` propagate to tool
    execution but NOT to model invocation — neither `on_chat_model_start`
    nor `on_llm_start` fires. Binding callbacks to the model via
    `model.with_config({"callbacks":[...]})` also does not surface them.
    So we cannot detect LLM-turn boundaries via callbacks at all.

    We sidestep this by relying on the `parallel_tool_calls=False` invariant
    enforced in `make_llm` / `bind_llm_for_question`: each LLM turn issues
    at most one tool call, so `n_tool_rounds == n_tool_calls`. The fields
    are kept distinct only for output-schema stability with the prior
    runs' log_record format.
    """

    def __init__(self, search_budget: int) -> None:
        super().__init__()
        self.n_tool_calls = 0
        self.n_search_calls = 0     # raw count, includes blocked attempts
        self.n_record_calls = 0
        self.n_tool_rounds = 0
        self.search_budget = search_budget

    async def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self.n_tool_calls += 1
        name = (serialized or {}).get("name", "")
        if name == "search_news":
            self.n_search_calls += 1
        elif name in ("record_candidates", "record_b16_answer"):
            self.n_record_calls += 1
        # parallel_tool_calls=False invariant — see class docstring.
        self.n_tool_rounds = self.n_tool_calls

    @property
    def search_budget_hit(self) -> bool:
        return self.n_search_calls > self.search_budget

    def snapshot(self) -> dict:
        return {
            "n_tool_rounds":     self.n_tool_rounds,
            "n_tool_calls":      self.n_tool_calls,
            "n_search_calls":    self.n_search_calls,
            "n_record_calls":    self.n_record_calls,
            "search_budget_hit": self.search_budget_hit,
        }


def _load_existing_predictions() -> tuple[list[dict], set[str]]:
    if not config.PREDICTIONS_PATH.exists():
        return [], set()
    preds: list[dict] = []
    done: set[str] = set()
    try:
        with config.PREDICTIONS_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                preds.append(rec)
                if "qid" in rec:
                    done.add(rec["qid"])
        return preds, done
    except Exception as e:
        print(f"Warning: existing predictions.jsonl unreadable ({e}); starting fresh")
        return [], set()


async def run_async(
    smoke: int | None,
    concurrency: int,
    benchmark_path=None,
    per_template_cap: int | None = None,
) -> None:
    bench = benchmark_path or config.BENCHMARK_PATH
    questions = load_benchmark(bench, per_template_cap=per_template_cap)
    print(f"Benchmark: {bench}")
    if per_template_cap is not None:
        print(f"per-template cap: {per_template_cap} → {len(questions)} questions total")
    if smoke:
        questions = questions[:smoke]
        print(f"SMOKE mode: first {smoke} questions only")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions, done_qids = _load_existing_predictions()
    pending = [q for q in questions if q["qid"] not in done_qids]

    print(f"Total target questions: {len(questions)}")
    print(f"Already in predictions.jsonl: {len(done_qids)} (skipped)")
    print(f"To process: {len(pending)} (concurrency={concurrency})")
    if not pending:
        print("Nothing to do.")
        return

    shared_client = make_shared_milvus_client()
    llm = make_llm()
    agent_sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    log_file = config.RUN_LOG_PATH.open("a")
    completed = len(done_qids)
    total = len(questions)
    t0 = time.time()

    async def process_and_write(q):
        nonlocal completed
        qid, prediction, log = await process_question(q, llm, shared_client, agent_sem)
        async with write_lock:
            predictions.append(prediction)
            line = format_jsonl_record(prediction["qid"], prediction["answer"])
            with config.PREDICTIONS_PATH.open("a") as f:
                f.write(line)
            log_file.write(json.dumps(log) + "\n")
            log_file.flush()
            completed += 1
            elapsed = time.time() - t0
            ntup = log.get("parsed_n_tuples", "ERR")
            nrounds = log.get("n_tool_rounds", "?")
            ncalls = log.get("n_tool_calls", "?")
            print(
                f"[{completed}/{total}] {qid} ({log.get('template_id')}) "
                f"parsed={ntup} rounds={nrounds} calls={ncalls} elapsed={elapsed:.0f}s",
                flush=True,
            )

    try:
        tasks = [asyncio.create_task(process_and_write(q)) for q in pending]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        log_file.close()

    n_exc = sum(1 for r in results if isinstance(r, Exception))
    elapsed = time.time() - t0
    print(f"\nAll done. {completed} predictions in predictions.jsonl. "
          f"Wall: {elapsed:.0f}s. Task exceptions: {n_exc}")
    print(f"  -> {config.PREDICTIONS_PATH}")
    print(f"  -> {config.RUN_LOG_PATH}")


def _redirect_run_outputs(name: str) -> None:
    """Point predictions/results/log at a sibling dir of outputs/."""
    new_dir = config.OUTPUTS_DIR.parent / name
    new_dir.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR = new_dir
    config.PREDICTIONS_PATH = new_dir / "predictions.jsonl"
    config.RESULTS_PATH = new_dir / "results.json"
    config.RUN_LOG_PATH = new_dir / "run_log.txt"
    config.RUN_NOTES_PATH = new_dir / "RUN_NOTES.md"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", type=int, default=None,
                   help="Run only first N questions")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="Max concurrent agent loops (default 5)")
    p.add_argument("--output-dir", default=None,
                   help="Sibling dir name for predictions/results/log "
                        "(default: outputs/). Use to keep multiple runs side-by-side.")
    p.add_argument("--benchmark", default=None,
                   help="Path to benchmark JSON. Defaults to config.BENCHMARK_PATH.")
    p.add_argument("--per-template-cap", type=int, default=None,
                   help="Cap questions per template_id (e.g. --per-template-cap 1 "
                        "with the cap50 alive set = 21-q smoke covering all templates). "
                        "Applied BEFORE --smoke when both are set.")
    args = p.parse_args()
    if args.output_dir:
        _redirect_run_outputs(args.output_dir)
    bench_path = None
    if args.benchmark:
        from pathlib import Path
        bp = Path(args.benchmark)
        bench_path = bp if bp.is_absolute() else (config.PROJECT_ROOT / bp)
    asyncio.run(run_async(smoke=args.smoke, concurrency=args.concurrency,
                          benchmark_path=bench_path,
                          per_template_cap=args.per_template_cap))


if __name__ == "__main__":
    main()
