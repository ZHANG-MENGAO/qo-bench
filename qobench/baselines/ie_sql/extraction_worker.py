#!/usr/bin/env python3
# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Cooperative worker for IE→SQL extraction on a PBS/SLURM batch system.

Multiple PBS jobs run this script in parallel. Coordination is via filesystem:
- `results/<article_id>.json` — written when extraction succeeds. Skip if exists with ok=true.
- `claims/<article_id>.claim` — atomic O_CREAT|O_EXCL lock. Holder processes the article.
  Stale claims (> CLAIM_TTL_SEC) are eligible for takeover.

In-process: ThreadPoolExecutor (THREADS env, default 30) processes claimed articles
in parallel against the local vLLM endpoint.

Per-worker shuffle seed (from PBS_JOBID) spreads initial claim attempts across articles
so workers don't all collide on the first task.

Logs go to LOG_PATH via tee'd stdout. The watchdog uses log mtime for alive check,
not PBS o-file which is spooled mid-run.

Usage (inside PBS job):
    python3 extraction_worker.py \
        --input input_articles.jsonl \
        --claims-dir claims \
        --results-dir results \
        --worker-id $PBS_JOBID
"""

import argparse, datetime, errno, json, os, random, re, sys, time
import threading, urllib.request, urllib.error
import concurrent.futures
from pathlib import Path

ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.6-27B-Instruct")
API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))
THREADS = int(os.environ.get("THREADS", "30"))
CLAIM_TTL_SEC = int(os.environ.get("CLAIM_TTL_SEC", "900"))  # 15 min

CHUNK_THRESHOLD = 6000
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 1000

# Prompt asset paths (alongside this script)
ASSETS = Path(__file__).parent
EXTRACTION_PROMPT_PATH = ASSETS / "extraction_prompt.md"
EVENT_DEFS_PATH = ASSETS / "event_definitions.md"
SCHEMA_PATH = ASSETS / "schema_v1_frozen.md"


def build_system_message() -> str:
    prompt_template = EXTRACTION_PROMPT_PATH.read_text()
    m = re.search(r"## System message \(used verbatim\)\n\n(.*?)\n\n## Implementation note",
                  prompt_template, re.DOTALL)
    if not m:
        sys.exit("ERROR: extraction_prompt.md missing System message section")
    sys_msg = m.group(1)
    event_defs = EVENT_DEFS_PATH.read_text()
    sys_msg = sys_msg.replace("{{EVENT_DEFINITIONS_MD}}", event_defs)
    schema_md = SCHEMA_PATH.read_text()
    m2 = re.search(r"```ts\n(.*?)\n```", schema_md, re.DOTALL)
    if not m2:
        sys.exit("ERROR: schema_v1_frozen.md missing ```ts``` block")
    schema_ts = "```typescript\n" + m2.group(1) + "\n```"
    sys_msg = sys_msg.replace("{{SCHEMA_TS}}", schema_ts)
    return sys_msg


def make_chunks(text: str):
    if len(text) <= CHUNK_THRESHOLD:
        return [(0, len(text), text)]
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + CHUNK_SIZE, len(text))
        chunks.append((pos, end, text[pos:end]))
        if end >= len(text):
            break
        pos = end - CHUNK_OVERLAP
    return chunks


def build_user_message(article, ci, n, cs, ce, ctext):
    h = (
        f"ARTICLE_ID: {article['article_id']}\n"
        f"ARTICLE_DATE: {article.get('article_date','')}\n"
        f"TITLE: {article.get('title','')}\n"
        f"URL: {article.get('url','')}\n"
    )
    if n > 1:
        h += (f"CHUNK: {ci+1} of {n} (body chars {cs}-{ce} of {len(article['text'])})\n"
              f"NOTE: This is part of a longer article. Extract events ONLY from the content below.\n"
              f"      Other parts of the article are processed separately; do not duplicate.\n")
    return h + "\nBODY:\n" + ctext


def call_vllm(system_msg, user_msg, temperature=TEMPERATURE, extra_system=""):
    msgs = [{"role": "system", "content": system_msg + extra_system},
            {"role": "user", "content": user_msg}]
    body = {"model": MODEL, "messages": msgs, "temperature": temperature, "max_tokens": MAX_TOKENS}
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            resp = json.loads(r.read())
        latency = time.time() - t0
    except Exception as e:
        return None, {}, time.time() - t0, f"{type(e).__name__}: {str(e)[:300]}"
    try:
        content = resp["choices"][0]["message"].get("content")
        usage = resp.get("usage", {}) or {}
    except (KeyError, IndexError, TypeError) as e:
        return None, {}, latency, f"ResponseParseError: {type(e).__name__}: {str(e)[:200]}"
    return content, usage, latency, None


def try_parse_json(s):
    if s is None:
        return None, "NoneContent"
    if not isinstance(s, str):
        return None, f"NonStringContent: type={type(s).__name__}"
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(json|typescript)?\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    try:
        return json.loads(s), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {str(e)[:200]}"


def extract_chunk(system_msg, article, ci, n, cs, ce, ctext):
    user_msg = build_user_message(article, ci, n, cs, ce, ctext)
    raw, usage, latency, err = call_vllm(system_msg, user_msg)
    if err:
        return {"chunk_idx": ci, "chunk_range": [cs, ce], "raw_output": None,
                "parsed_events": None, "call_error": err, "parse_error": None,
                "latency_s": round(latency, 2), "usage": usage, "retried": False}
    parsed, perr = try_parse_json(raw)
    if parsed is not None:
        return {"chunk_idx": ci, "chunk_range": [cs, ce], "raw_output": raw,
                "parsed_events": parsed, "call_error": None, "parse_error": None,
                "latency_s": round(latency, 2), "usage": usage, "retried": False}
    # Retry once
    extra = "\n\nIMPORTANT: Output MUST be a single JSON array, parseable by json.loads. No markdown code fences, no prose."
    raw2, usage2, latency2, err2 = call_vllm(system_msg, user_msg, temperature=0.2, extra_system=extra)
    latency += latency2
    for k in ("prompt_tokens", "completion_tokens"):
        usage[k] = usage.get(k, 0) + usage2.get(k, 0)
    parsed2, perr2 = (None, err2) if err2 else try_parse_json(raw2)
    return {"chunk_idx": ci, "chunk_range": [cs, ce], "raw_output": raw,
            "raw_output_retry": raw2, "parsed_events": parsed2,
            "call_error": err2, "parse_error": perr2 if parsed2 is None else None,
            "latency_s": round(latency, 2), "usage": usage, "retried": True}


def process_article(system_msg, article, worker_id):
    chunks_list = make_chunks(article["text"])
    n = len(chunks_list)
    chunk_results = []
    for ci, (cs, ce, ctext) in enumerate(chunks_list):
        chunk_results.append(extract_chunk(system_msg, article, ci, n, cs, ce, ctext))
    all_events = []
    for cr in chunk_results:
        if cr.get("parsed_events"):
            for e in cr["parsed_events"]:
                if isinstance(e, dict):
                    e["_chunk_idx"] = cr["chunk_idx"]
                    all_events.append(e)
    total_lat = sum(cr["latency_s"] for cr in chunk_results)
    total_in = sum(cr["usage"].get("prompt_tokens", 0) for cr in chunk_results)
    total_out = sum(cr["usage"].get("completion_tokens", 0) for cr in chunk_results)
    return {
        "article_id": article["article_id"],
        "article_date": article.get("article_date"),
        "title": article.get("title"),
        "body_chars": len(article["text"]),
        "n_chunks": n,
        "chunks": chunk_results,
        "parsed_events_all": all_events,
        "n_events": len(all_events),
        "n_parse_errors": sum(1 for cr in chunk_results if cr.get("parse_error")),
        "n_call_errors": sum(1 for cr in chunk_results if cr.get("call_error")),
        "n_retried": sum(1 for cr in chunk_results if cr.get("retried")),
        "total_latency_s": round(total_lat, 2),
        "usage": {"prompt_tokens": total_in, "completion_tokens": total_out},
        "processed_at": datetime.datetime.now().isoformat(),
        "worker_id": worker_id,
        "ok": True,
    }


def is_done(result_path: Path) -> bool:
    if not result_path.exists():
        return False
    try:
        r = json.loads(result_path.read_text())
        return r.get("ok") is True
    except Exception:
        return False


def try_claim(claim_path: Path, worker_id: str) -> bool:
    """Atomic claim. Returns True if we got it. Handles stale-claim takeover."""
    payload = json.dumps({"worker": worker_id, "ts": time.time()})
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        return True
    except FileExistsError:
        # Check if stale
        try:
            existing = json.loads(claim_path.read_text())
            age = time.time() - existing.get("ts", 0)
            if age > CLAIM_TTL_SEC:
                # Take over
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
                # Retry claim
                try:
                    fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    with os.fdopen(fd, "w") as f:
                        f.write(payload)
                    return True
                except FileExistsError:
                    return False
        except Exception:
            pass
        return False


def release_claim(claim_path: Path):
    try:
        claim_path.unlink()
    except FileNotFoundError:
        pass


print_lock = threading.Lock()
def log(msg):
    with print_lock:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def process_one(system_msg, article, claims_dir, results_dir, worker_id, stats):
    aid = article["article_id"]
    rpath = results_dir / f"{aid}.json"
    cpath = claims_dir / f"{aid}.claim"
    if is_done(rpath):
        stats["skipped_done"] += 1
        return
    if not try_claim(cpath, worker_id):
        stats["skipped_claimed"] += 1
        return
    try:
        result = process_article(system_msg, article, worker_id)
        rpath.write_text(json.dumps(result, ensure_ascii=False))
        stats["processed"] += 1
        if stats["processed"] % 25 == 0:
            log(f"  processed={stats['processed']}  skip_done={stats['skipped_done']}  skip_claimed={stats['skipped_claimed']}")
    except Exception as e:
        log(f"  ERROR {aid}: {type(e).__name__}: {e}")
        stats["errors"] += 1
        # Write error result so future workers know this is a problem
        try:
            rpath.write_text(json.dumps({"article_id": aid, "ok": False,
                                          "error": f"{type(e).__name__}: {str(e)[:300]}",
                                          "worker_id": worker_id,
                                          "processed_at": datetime.datetime.now().isoformat()}, ensure_ascii=False))
        except Exception:
            pass
    finally:
        release_claim(cpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--claims-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--worker-id", required=True)
    args = ap.parse_args()

    log(f"Worker {args.worker_id} starting")
    log(f"  endpoint={ENDPOINT}  model={MODEL}  threads={THREADS}  claim_ttl={CLAIM_TTL_SEC}s")

    claims_dir = Path(args.claims_dir); claims_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    system_msg = build_system_message()
    log(f"  system message: {len(system_msg):,} chars")

    articles = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    log(f"  input articles: {len(articles):,}")

    # Per-worker shuffle to spread initial claims
    seed_int = sum(ord(c) for c in args.worker_id)
    random.seed(seed_int)
    random.shuffle(articles)
    log(f"  shuffled with seed={seed_int} (from worker_id)")

    stats = {"processed": 0, "skipped_done": 0, "skipped_claimed": 0, "errors": 0}
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = [ex.submit(process_one, system_msg, a, claims_dir, results_dir, args.worker_id, stats)
                for a in articles]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                log(f"  thread error: {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    log(f"Worker {args.worker_id} done in {elapsed/60:.1f}m: processed={stats['processed']} "
        f"skip_done={stats['skipped_done']} skip_claimed={stats['skipped_claimed']} errors={stats['errors']}")


if __name__ == "__main__":
    main()
