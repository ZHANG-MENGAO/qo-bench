#!/usr/bin/env python3
# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""
Run GraphRAG queries on the QO-Bench question set.

- Reads benchmark/questions/questions.jsonl.gz (785 q × 18 templates).
- For each (qid, method) in (qids × {local, global, drift}):
  - Skip if a successful parsed result already exists.
  - Build per-template response-type asking for STRICT JSON.
  - Invoke `graphrag query`; capture stdout (content only; thinking already
    routed to reasoning_content via vLLM's --reasoning-parser qwen3).
  - Parse JSON. On failure retry up to MAX_RETRIES with stricter prompt.
  - Save raw + parsed + per-attempt errors to results/{method}/{qid}.json.

- Then consolidate into predictions_{method}.jsonl matching eval.py format.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------- Config ----------

ROOT = Path(__file__).resolve().parent              # graphrag project root (settings.yaml + input/ + output/ live here)
BUNDLE_ROOT = Path(__file__).resolve().parents[3]   # supplementary bundle root
DEFAULT_QUESTIONS = BUNDLE_ROOT / "benchmark" / "questions" / "questions.jsonl.gz"
DEFAULT_RESULTS_DIR = ROOT / "eval_results"

DEFAULT_METHODS = ["local", "global", "drift"]
COMMUNITY_LEVEL = 2  # default per GraphRAG paper; global only
MAX_RETRIES = 3
PER_QUERY_TIMEOUT = 1800  # 30 min per query (drift is slowest)
CLAIM_TTL_SEC = 1800      # 30 min: if claim mtime older than this, treat as stale

# Set at runtime from CLI / env
QUESTIONS: Path = DEFAULT_QUESTIONS
RESULTS_DIR: Path = DEFAULT_RESULTS_DIR
WORKER_ID: str = os.environ.get("PBS_JOBID", str(os.getpid())).split(".")[0]

# Per-template response-type (appended to LLM prompt via --response-type)
RT_LIST = (
    "Return ONLY a JSON array (no prose, no markdown fences, no leading/trailing text). "
    "STRICT FORMAT RULES: "
    "(1) firm_ticker / buyer_ticker / target_ticker MUST be uppercase stock ticker symbols "
    "(e.g. AAPL, MSFT, TWTR) -- NEVER company names like Microsoft or Twitter. "
    "(2) ALL date fields (anchor_date, announce_date, close_date, e1_anchor_date, "
    "e2_anchor_date, etc.) MUST be exact YYYY-MM-DD; "
    "do NOT output partial dates like 2013, November 2013, or late 2013 -- "
    "pick the most specific date found in the source. "
    "(3) cited_urls MUST be actual article URLs taken verbatim from article_url: "
    "lines in the retrieved source content -- NEVER source IDs, Reports (N), "
    "or Sources (N) references. "
    "Each element is an object with the following fields:"
)

RESPONSE_TYPE = {
    "A.1.1": f"{RT_LIST} firm_ticker (string), event_type (string), anchor_date (YYYY-MM-DD), cited_urls (list of article URLs supporting this row). Empty array [] if no matching events.",
    "A.1.2": f"{RT_LIST} firm_ticker (string), cited_urls (list of URLs). Distinct firms only.",
    "A.2.1": f"{RT_LIST} firm_ticker, event_type, anchor_date (YYYY-MM-DD), cited_urls.",
    "A.3.1": f"{RT_LIST} firm_ticker, event_type, anchor_date, cited_urls. For Entity-projection (T18) form: only firm_ticker per row. For Event-projection (T40) form: include event_type and anchor_date too.",
    "A.3.combined": f"{RT_LIST} firm_ticker, cited_urls. (The role / event-type filter was specified in the question.)",
    "B.1.1": f"{RT_LIST} firm_ticker, event_type, anchor_date, cited_urls. List events occurring before the anchor.",
    "B.1.2": f"{RT_LIST} firm_ticker, e1_anchor_date (YYYY-MM-DD), e2_anchor_date, gap_days (int), cited_urls.",
    "B.1.3": f"{RT_LIST} firm_ticker, e_trigger_anchor_date, e_followup_anchor_date, lag_days (int), cited_urls.",
    "B.1.4": f"{RT_LIST} buyer_ticker, target_ticker, announce_date, close_date, cited_urls.",
    "B.1.5": "Return ONLY a single integer (number of days between the two events). No prose, no JSON wrapper, just the number.",
    "B.2.1": f"{RT_LIST} firm_ticker, event_type, anchor_date, cited_urls. Ordered chronologically (earliest/latest per question wording).",
    "B.2.2": f"{RT_LIST} firm_ticker, event_type, anchor_date, cited_urls. Ordered chronologically by anchor_date.",
    "B.3.1": f"{RT_LIST} firm_ticker, cited_urls. Firms with BOTH event types.",
    "B.3.2": f"{RT_LIST} firm_ticker, cited_urls. Firms appearing in BOTH windows.",
    "B.3.3": f"{RT_LIST} firm_ticker, cited_urls. Firms in BOTH event-type sets.",
    "B.4.1": f"{RT_LIST} firm_ticker, cited_urls. Firms with >= N events as specified.",
    "B.4.2": f"{RT_LIST} firm_ticker, cited_urls. Firms in role >= N times.",
    "B.4.3": f"{RT_LIST} firm_ticker, event_type, anchor_date, bucket (string label like '2024-Q1'), cited_urls.",
    "B.4.4": f"{RT_LIST} firm_ticker, event_type, anchor_date, cited_urls. event_type field is the type label per row.",
}

STRICTER_PREFIX = (
    "PREVIOUS ATTEMPT RETURNED INVALID JSON. You MUST output only valid JSON. "
    "No code fences. No 'Here is...' preamble. No 'Note:' postamble. "
    "Start the response with '[' or a digit. Try again. "
)


# ---------- Helpers ----------

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def parse_response(content: str, tpl_id: str):
    """Returns (parsed_answer, error_string_or_None)."""
    content = content.strip()
    # Strip common code-fence wrappers
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        content = content.strip()
    # int_match special (B.1.5 = composite_b16, scalar integer answer)
    if tpl_id == "B.1.5":
        m = re.search(r"-?\d+", content)
        if m:
            return int(m.group()), None
        return None, "no integer found"
    # Try direct JSON parse
    try:
        ans = json.loads(content)
        if isinstance(ans, list):
            return ans, None
        return None, f"top-level is {type(ans).__name__}, expected list"
    except json.JSONDecodeError as e:
        pass
    # Fallback: extract first balanced JSON array
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if m:
        try:
            ans = json.loads(m.group())
            if isinstance(ans, list):
                return ans, None
        except json.JSONDecodeError:
            pass
    return None, "no valid JSON array found"


# ---------- Claim / lease for multi-worker coordination ----------

def _claim_path(out_path: Path) -> Path:
    return out_path.with_suffix(".claim")


def try_claim(out_path: Path) -> bool:
    """Atomically try to claim this qid.

    Returns True if we own the claim and should run the query.
    Returns False if another live worker has it.
    """
    cp = _claim_path(out_path)
    payload = f"{WORKER_ID}\n{int(time.time())}\n"
    # Atomic create (O_CREAT | O_EXCL)
    try:
        fd = os.open(str(cp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        return True
    except FileExistsError:
        pass
    # Check if existing claim is stale
    try:
        lines = cp.read_text().splitlines()
        ts = int(lines[1])
    except Exception:
        ts = 0
    if time.time() - ts > CLAIM_TTL_SEC:
        # Stale — take over. Race window is small.
        try:
            cp.unlink()
        except FileNotFoundError:
            pass
        try:
            fd = os.open(str(cp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            return True
        except FileExistsError:
            return False  # someone else just grabbed it
    return False


def release_claim(out_path: Path):
    try:
        _claim_path(out_path).unlink()
    except FileNotFoundError:
        pass


def call_graphrag(method: str, query: str, response_type: str) -> tuple[str, str, int]:
    # graphrag query takes the query as a POSITIONAL argument (not --query).
    # We build the option list first then append the positional last.
    cmd = [
        "python", "-m", "graphrag", "query",
        "--root", str(ROOT),
        "--method", method,
        "--response-type", response_type,
    ]
    if method == "global":
        cmd += ["--community-level", str(COMMUNITY_LEVEL)]
    cmd += [query]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PER_QUERY_TIMEOUT)
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        return (e.stdout.decode() if e.stdout else ""), f"TIMEOUT after {PER_QUERY_TIMEOUT}s", -1


def run_one(qid: str, tpl_id: str, nl_question: str, method: str, out_path: Path) -> dict:
    attempts = []
    answer = None
    last_stdout = ""
    last_stderr = ""
    for attempt in range(1, MAX_RETRIES + 1):
        rt = RESPONSE_TYPE.get(tpl_id, f"{RT_LIST} relevant fields and cited_urls.")
        if attempt > 1:
            rt = STRICTER_PREFIX + rt
        t0 = time.time()
        stdout, stderr, rc = call_graphrag(method, nl_question, rt)
        elapsed = time.time() - t0
        last_stdout = stdout
        last_stderr = stderr
        ans, err = parse_response(stdout, tpl_id)
        attempts.append({
            "attempt": attempt,
            "elapsed_s": round(elapsed, 1),
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
            "rc": rc,
            "parse_error": err,
        })
        if err is None:
            answer = ans
            break

    result = {
        "qid": qid,
        "id": tpl_id,
        "method": method,
        "answer": answer,
        "ok": answer is not None,
        "attempts": attempts,
        "raw_last_stdout": last_stdout[-4000:],
        "raw_last_stderr": last_stderr[-4000:],
    }
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def main():
    global QUESTIONS, RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help="Comma-separated subset of {local,global,drift}.")
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS),
                    help="Path to questions jsonl.gz")
    ap.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR),
                    help="Where to write per-qid results + predictions_<method>.jsonl")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="If >0, shuffle questions with this seed (helps distribute work "
                         "across workers when claim collisions are common).")
    args = ap.parse_args()

    QUESTIONS = Path(args.questions)
    RESULTS_DIR = Path(args.output_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in DEFAULT_METHODS]
    if bad:
        log(f"Unknown methods: {bad}; valid = {DEFAULT_METHODS}")
        sys.exit(2)
    log(f"Worker: {WORKER_ID}  Methods: {methods}  Shuffle seed: {args.shuffle_seed}")
    log(f"Questions: {QUESTIONS}")
    log(f"Output dir: {RESULTS_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for m in methods:
        (RESULTS_DIR / m).mkdir(exist_ok=True)

    # Load questions
    questions = []
    with gzip.open(QUESTIONS, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    log(f"Loaded {len(questions)} questions")

    # Optional shuffle per-worker (reduces initial collisions on shared work queue)
    if args.shuffle_seed:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(questions)
        log(f"Shuffled questions with seed {args.shuffle_seed}")

    total = len(questions) * len(methods)
    counted = 0
    fail = 0
    skipped_already_ok = 0
    skipped_claimed = 0
    for q in questions:
        qid = q["qid"]
        tpl_id = q["id"]
        nl = q["nl_question"]
        for method in methods:
            counted += 1
            out_path = RESULTS_DIR / method / f"{qid}.json"
            # 1. Skip if already OK
            if out_path.exists():
                try:
                    existing = json.loads(out_path.read_text())
                    if existing.get("ok"):
                        skipped_already_ok += 1
                        continue
                except Exception:
                    pass
            # 2. Try to claim (multi-worker coordination)
            if not try_claim(out_path):
                skipped_claimed += 1
                continue
            # 3. Process
            t0 = time.time()
            log(f"[{counted}/{total}] {method} {tpl_id} {qid[:8]} ...")
            try:
                r = run_one(qid, tpl_id, nl, method, out_path)
            finally:
                release_claim(out_path)
            elapsed = time.time() - t0
            status = "OK" if r["ok"] else "FAIL"
            log(f"  -> {status} after {len(r['attempts'])} attempts ({elapsed:.0f}s)")
            if not r["ok"]:
                fail += 1

    log(f"\nWorker {WORKER_ID} done. checked={counted} "
        f"skipped_done={skipped_already_ok} skipped_claimed={skipped_claimed} fail={fail}")

    log(f"\nDONE. {done}/{total} queries ran. {fail} failed after retries.")

    # Consolidate per-method predictions
    for method in methods:
        preds = []
        for f in sorted((RESULTS_DIR / method).glob("*.json")):
            r = json.loads(f.read_text())
            ans = r.get("answer")
            if ans is None:
                # eval.py needs a value; use [] as sentinel (B.1.5 is the scalar-int composite)
                ans = [] if r["id"] != "B.1.5" else 0
            preds.append({"qid": r["qid"], "answer": ans})
        out = RESULTS_DIR / f"predictions_{method}.jsonl"
        with out.open("w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        log(f"  wrote {out.name} ({len(preds)} entries)")


if __name__ == "__main__":
    main()
