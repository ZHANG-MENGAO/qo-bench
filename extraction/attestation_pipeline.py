# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-NC-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""
Coverage pipeline: for each KeyDev M&A Announcement event in a date range,
determine whether the news corpus (db.mna) covers it.

L0 = any article linked by event_id (count)
L1 = among those articles, how many have participant company name in title
L2 = single-judge LLM verdict (early-stop: covered as soon as one article confirmed)

The paper's 3-judge attestation runs this funnel once per judge model and
keeps a pair only on unanimous 3-of-3 confirmation. Select the judge via the
LLM_MODEL env var (the three judges are Gemma-4-31B-IT, Qwen3.6-27B, and
gpt-oss-120B; see ../docs/IE_PIPELINE.md).

Outputs per-event JSON to <data-root>/coverage_pipeline/<bucket>/<event_id>.json

Reviewer note: This script originally targeted an internal MongoDB collection
of KeyDev-linked news articles. To re-run, set MONGO_URI to your own Mongo
deployment with a compatible collection (see ../docs/IE_PIPELINE.md for
schema), and LLM_URL to your own vLLM/OpenAI-compatible chat endpoint. The
original is not redistributed because it points to proprietary upstream
data (S&P Capital IQ KeyDev events).
"""
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from pymongo import MongoClient

MONGO_URI = os.environ.get(
    "MONGO_URI",
    "<set MONGO_URI env var; format: mongodb://user:pass@host:port/db>",
)
LLM_URL = os.environ.get(
    "LLM_URL",
    "<set LLM_URL env var; format: https://your-vllm-host:port/v1/chat/completions>",
)
LLM_MODEL = os.environ.get("LLM_MODEL", "Gemma-4-31B-IT")  # judge model; run once per judge for 3-of-3
CONCURRENCY = int(os.environ.get("CONCURRENCY", "12"))     # concurrent in-flight LLM requests
TOP_K = 10         # L2 judges top K articles by |days_to_event|
EXCERPT_CHARS = 800

LEGAL_SUFFIXES = [
    " incorporated", " corporation", " company", " holdings", " group",
    " limited", " inc.", " inc", " corp.", " corp", " ltd.", " ltd",
    " llc", " l.l.c.", " llp", " lp", " l.p.", " plc", " p.l.c.",
    " s.a.", " sa", " se", " ag", " n.v.", " nv", " b.v.", " bv",
    " pty", " pty.", " holdings", " international",
]
STOPWORD_FIRST_TOKENS = {
    "the", "new", "american", "united", "global", "national",
    "first", "general", "national", "international", "an", "a",
}


def normalize_name(name: str) -> set:
    """Return a set of searchable normalized forms of a company name."""
    if not name:
        return set()
    n = name.lower()
    n = re.sub(r"\([^)]*\)", "", n)
    n = n.split(",")[0]
    n = re.sub(r"\s+", " ", n).strip()
    for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        while n.endswith(s):
            n = n[: -len(s)].strip()
    n = n.strip(" .,")
    out = set()
    if len(n) >= 3:
        out.add(n)
    tokens = n.split()
    if len(tokens) >= 2:
        out.add(" ".join(tokens[:2]))
    if tokens:
        first = tokens[0]
        if len(first) >= 4 and first not in STOPWORD_FIRST_TOKENS:
            out.add(first)
    return {x for x in out if len(x) >= 3}


def extract_participant_names(participants: list) -> dict:
    """Return dict of role→list of {ticker, sp_companyid, name, normalized_names}."""
    out = {}
    for p in participants:
        role = p.get("role") or "?"
        item = {
            "ticker": p.get("ticker"),
            "sp_companyid": p.get("sp_companyid"),
            "name": p.get("company_name"),
            "normalized": list(normalize_name(p.get("company_name") or "")),
        }
        out.setdefault(role, []).append(item)
    return out


def l1_title_match(title: str, all_names: set) -> list:
    """Return list of names found in title."""
    if not title:
        return []
    t = title.lower()
    hits = []
    for n in all_names:
        if n in t and len(n) >= 3:
            hits.append(n)
    return hits


def build_l2_prompt(event_hl: str, buyer_name: str, target_name: str, anchor_date: str,
                    article_title: str, article_date: str, article_excerpt: str) -> list:
    system = (
        "You are judging whether a news article is reporting on a specific M&A deal. "
        "Answer with exactly one token: YES or NO."
    )
    user = (
        f"M&A deal being investigated:\n"
        f"  Announcement date: {anchor_date}\n"
        f"  Buyer: {buyer_name or 'unknown'}\n"
        f"  Target: {target_name or 'unknown'}\n"
        f"  Original KeyDev headline: {event_hl}\n\n"
        f"Candidate article:\n"
        f"  Date: {article_date}\n"
        f"  Title: {article_title}\n"
        f"  Excerpt (first {EXCERPT_CHARS} chars): {article_excerpt}\n\n"
        f"Does this article primarily report on the specific M&A deal described above? "
        f"(Not just mention either company in passing.) Answer exactly YES or NO."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def call_llm(client: httpx.AsyncClient, messages: list, sem: asyncio.Semaphore) -> str:
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(
                    LLM_URL,
                    json={
                        "model": LLM_MODEL,
                        "messages": messages,
                        "max_tokens": 4,
                        "temperature": 0.0,
                    },
                    timeout=60.0,
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"].strip().upper()
            except Exception as e:
                if attempt == 2:
                    return f"ERROR:{type(e).__name__}"
                await asyncio.sleep(1.5 * (attempt + 1))
        return "ERROR:retry_exhausted"


async def judge_event_l2(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                          event: dict, articles: list, verbose=False) -> dict:
    """Run L2 on top-K articles in parallel; early-return if any says YES."""
    participants = event["participants_parsed"]
    buyer = participants.get("Buyer", [{}])[0].get("name", "")
    target = participants.get("Target", [{}])[0].get("name", "")
    headline = event.get("headline", "")
    anchor = event.get("anchor_date_str", "")

    top = articles[:TOP_K]
    if not top:
        return {"verdict": "absent", "l2_sample": [], "l2_yes": 0, "l2_no": 0, "l2_error": 0}

    async def judge_one(art):
        excerpt = (art.get("article_text") or "")[:EXCERPT_CHARS]
        messages = build_l2_prompt(
            headline, buyer, target, anchor,
            art.get("article_title") or "",
            art.get("article_date_str") or "",
            excerpt,
        )
        verdict = await call_llm(client, messages, sem)
        return {
            "url": art.get("article_url"),
            "date": art.get("article_date_str"),
            "title": (art.get("article_title") or "")[:120],
            "days_to_event": art.get("days_to_event"),
            "verdict": verdict,
        }

    results = await asyncio.gather(*[judge_one(a) for a in top])
    n_yes = sum(1 for r in results if r["verdict"].startswith("YES"))
    n_no = sum(1 for r in results if r["verdict"].startswith("NO"))
    n_err = sum(1 for r in results if r["verdict"].startswith("ERROR"))

    if n_yes >= 1:
        final = "covered"
    elif n_no >= (TOP_K - n_err):
        final = "absent"
    else:
        final = "uncertain"

    return {
        "verdict": final,
        "l2_sample": results,
        "l2_yes": n_yes,
        "l2_no": n_no,
        "l2_error": n_err,
    }


async def process_bucket(bucket_name: str, date_lo: datetime, date_hi: datetime,
                          out_dir: Path):
    print(f"\n=== Processing bucket: {bucket_name}  ({date_lo.date()} to {date_hi.date()}) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    mc = MongoClient(MONGO_URI)
    idx = mc.news_corpus.events_index
    mna = mc.news_corpus.mna

    events = list(idx.find({
        "event_type_id": 80,
        "anchor_date": {"$gte": date_lo, "$lt": date_hi},
    }).sort("anchor_date", 1))
    print(f"  events pulled: {len(events)}")

    # Pre-process events + fetch articles
    prepped = []
    for ev in events:
        eid = ev["event_id"]
        anchor = ev["anchor_date"]
        parts = extract_participant_names(ev.get("participants", []))
        # Flatten all searchable names across roles
        all_names = set()
        for role, items in parts.items():
            for it in items:
                all_names.update(it["normalized"])

        # Articles
        arts_cur = mna.find({"event_id": eid}).sort("article_date", 1)
        arts = []
        for a in arts_cur:
            ad = a.get("article_date")
            arts.append({
                "article_url": a.get("article_url"),
                "article_title": a.get("article_title"),
                "article_text": a.get("article_text") or "",
                "article_date": ad,
                "article_date_str": ad.strftime("%Y-%m-%d") if ad else None,
                "days_to_event": a.get("days_to_event"),
                "role": a.get("role"),
                "ticker": a.get("ticker"),
            })
        # Sort by |days_to_event| asc for top-K selection; fallback to date proximity
        def sort_key(a):
            dte = a.get("days_to_event")
            if dte is None and a.get("article_date") and anchor:
                dte = (a["article_date"] - anchor).days
            return abs(dte) if dte is not None else 99999
        arts.sort(key=sort_key)

        # L1 compute (over all articles, not just top-K)
        l1_hits = []
        for a in arts:
            hit_names = l1_title_match(a.get("article_title") or "", all_names)
            if hit_names:
                l1_hits.append({
                    "url": a["article_url"],
                    "date": a["article_date_str"],
                    "title": (a.get("article_title") or "")[:120],
                    "hit_names": hit_names,
                    "days_to_event": a.get("days_to_event"),
                })

        prepped.append({
            "event_id": eid,
            "anchor_date": anchor,
            "anchor_date_str": anchor.strftime("%Y-%m-%d") if anchor else None,
            "event_type_id": ev.get("event_type_id"),
            "event_category": ev.get("event_category"),
            "n_participants": ev.get("n_participants"),
            "participants_parsed": parts,
            "headline": (ev.get("keydev_source") or {}).get("headline"),
            "articles": arts,
            "l0_count": len(arts),
            "l1_count": len(l1_hits),
            "l1_hits": l1_hits,
            "all_names": sorted(all_names),
        })

    # L2 pass
    sem = asyncio.Semaphore(CONCURRENCY)
    # TLS verification follows httpx defaults. Set LLM_VERIFY_TLS=0 only if you
    # must talk to a self-signed endpoint you control.
    _verify_tls = os.environ.get("LLM_VERIFY_TLS", "1") != "0"
    async with httpx.AsyncClient(verify=_verify_tls) as client:
        tasks = []
        for ev in prepped:
            if ev["l0_count"] == 0:
                # skip LLM, mark absent
                ev["l2"] = {"verdict": "absent", "l2_sample": [],
                            "l2_yes": 0, "l2_no": 0, "l2_error": 0}
                tasks.append(asyncio.sleep(0))
            else:
                async def run(evx=ev):
                    evx["l2"] = await judge_event_l2(client, sem, evx, evx["articles"])
                tasks.append(asyncio.create_task(run()))
        # progress
        done_n = 0
        for t in asyncio.as_completed(tasks):
            await t
            done_n += 1
            if done_n % 10 == 0 or done_n == len(tasks):
                print(f"  L2 progress: {done_n}/{len(tasks)}")

    # Write per-event JSON
    for ev in prepped:
        # Final verdict: L2 > L1 > L0
        if ev["l0_count"] == 0:
            final = "absent"
        elif ev["l2"]["verdict"] == "covered":
            final = "covered"
        elif ev["l2"]["verdict"] == "uncertain":
            final = "uncertain"
        else:
            final = "absent_per_l2"

        record = {
            "bucket": bucket_name,
            "event_id": ev["event_id"],
            "anchor_date": ev["anchor_date_str"],
            "event_type_id": ev["event_type_id"],
            "event_category": ev["event_category"],
            "n_participants": ev["n_participants"],
            "headline": ev["headline"],
            "participants": ev["participants_parsed"],
            "all_normalized_names": ev["all_names"],
            "l0_count": ev["l0_count"],
            "l1_count": ev["l1_count"],
            "l1_hits": ev["l1_hits"][:20],  # cap at 20 entries in JSON
            "l2": ev["l2"],
            "final_verdict": final,
        }
        out_path = out_dir / f"{ev['event_id']}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))

    # Summary for this bucket
    from collections import Counter
    verdicts = Counter(("absent" if e["l0_count"] == 0 else e["l2"]["verdict"]) for e in prepped)
    l0_pos = sum(1 for e in prepped if e["l0_count"] > 0)
    l1_pos = sum(1 for e in prepped if e["l1_count"] > 0)
    l2_pos = sum(1 for e in prepped if e["l2"]["verdict"] == "covered")
    print(f"\n  Bucket summary: {bucket_name}")
    print(f"    total events      : {len(prepped)}")
    print(f"    L0 > 0 (has any article)     : {l0_pos}")
    print(f"    L1 > 0 (any title name match): {l1_pos}")
    print(f"    L2 covered (LLM confirms ≥1) : {l2_pos}")
    print(f"    verdicts: {dict(verdicts)}")
    return prepped


async def main():
    out_base = Path("<data-root>/coverage_pipeline")
    out_base.mkdir(parents=True, exist_ok=True)

    # Q1 2023 test set
    q1_2023 = await process_bucket(
        "q1_2023",
        datetime(2023, 1, 1), datetime(2023, 4, 1),
        out_base / "q1_2023",
    )
    # Q1 2018 control
    q1_2018 = await process_bucket(
        "q1_2018",
        datetime(2018, 1, 1), datetime(2018, 4, 1),
        out_base / "q1_2018",
    )


if __name__ == "__main__":
    asyncio.run(main())
