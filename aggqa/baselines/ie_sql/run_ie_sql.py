#!/usr/bin/env python3
# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""IE→SQL paradigm runner: template-conditioned SQL with param injection.

Reads questions JSONL, dispatches each to a SQL family based on the paper
template id (A/B scheme), fills typed params into the SQL skeleton, executes
against events.duckdb, formats answer to match template's output_schema,
writes predictions JSONL ready for eval.py / eval_tolerant.py.

Usage:
    python3 run_ie_sql.py \\
        --db events.duckdb \\
        --questions ../../../benchmark/questions/questions.jsonl.gz \\
        --output predictions.jsonl
"""

import argparse, gzip, json, sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import duckdb


# Benchmark "Buyer/Target/Seller" → our DB junction role
ROLE_MAP = {
    "Buyer":  "acquirer",
    "Target": "target",
    "Seller": "seller",
}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def fmt_date(d):
    if d is None:
        return None
    if isinstance(d, str):
        return d[:10]
    return d.isoformat()


def get_cited_urls(con, event_ids):
    """Return list of article URLs for an event_id (or list of event_ids)."""
    if isinstance(event_ids, str):
        event_ids = [event_ids]
    if not event_ids:
        return []
    placeholders = ",".join(["?"] * len(event_ids))
    rows = con.execute(
        f"SELECT DISTINCT article_url FROM event_sources WHERE event_id IN ({placeholders}) AND article_url IS NOT NULL",
        event_ids,
    ).fetchall()
    return [r[0] for r in rows]


def firm_filter_clause(table_alias="e", firm_field="firm_ticker"):
    """SQL fragment matching a firm by ticker (primary). For M&A, also via roles."""
    return f"({table_alias}.{firm_field} = ?)"


def firm_in_event_sql(firm_param_idx=1):
    """Returns SQL fragment: event_id is for a 'firm' in any role (primary/acquirer/target/seller/issuer)."""
    return """
      e.event_id IN (
        SELECT event_id FROM event_companies
        WHERE ticker = ? OR normalized_name = LOWER(REPLACE(?, ' ', '_'))
      )
    """


# ────────────────────────────────────────────────────────────────────────────
# SQL families per template id
# ────────────────────────────────────────────────────────────────────────────

def run_A11(con, p):
    """A.1.1 — type+window → events. identity: firm_ticker, event_type, anchor_date."""
    rows = con.execute("""
        SELECT firm_ticker, event_type, event_date_value, event_id
        FROM events
        WHERE event_type = ?
          AND event_date_value BETWEEN ? AND ?
          AND firm_ticker IS NOT NULL
    """, [p["E"], p["W_start"], p["W_end"]]).fetchall()
    return [
        {
            "firm_ticker": r[0],
            "event_type": r[1],
            "anchor_date": fmt_date(r[2]),
            "cited_urls": get_cited_urls(con, [r[3]]),
        }
        for r in rows
    ]


def run_A12(con, p):
    """A.1.2 — type+window → distinct firms. identity: firm_ticker."""
    rows = con.execute("""
        SELECT firm_ticker, MIN(event_id), MAX(event_id)
        FROM events
        WHERE event_type = ?
          AND event_date_value BETWEEN ? AND ?
          AND firm_ticker IS NOT NULL
        GROUP BY firm_ticker
    """, [p["E"], p["W_start"], p["W_end"]]).fetchall()
    out = []
    for r in rows:
        evs = list({r[1], r[2]})
        out.append({"firm_ticker": r[0], "cited_urls": get_cited_urls(con, evs)})
    return out


def run_A21(con, p):
    """A.2.1 — type+firm+window → events with role/counterparty for M&A.
    Firm participation: primary OR via M&A role (acquirer/target/seller)."""
    F = p["F"]
    rows = con.execute(f"""
        WITH firm_events AS (
          SELECT DISTINCT event_id FROM event_companies
          WHERE ticker = ? OR UPPER(ticker) = UPPER(?)
        )
        SELECT e.firm_ticker, e.event_type, e.event_date_value, e.event_id
        FROM events e
        WHERE e.event_type = ?
          AND e.event_date_value BETWEEN ? AND ?
          AND (e.firm_ticker = ? OR e.event_id IN (SELECT event_id FROM firm_events))
    """, [F, F, p["E"], p["W_start"], p["W_end"], F]).fetchall()
    out = []
    for r in rows:
        firm_ticker = r[0] or F
        out.append({
            "firm_ticker": firm_ticker,
            "event_type": r[1],
            "anchor_date": fmt_date(r[2]),
            "cited_urls": get_cited_urls(con, [r[3]]),
        })
    return out


def run_A31(con, p):
    """A.3.1 (v18: T18+T40 merged) — type+role+window → events with firm_ticker."""
    R = ROLE_MAP.get(p["R"], p["R"].lower())
    rows = con.execute(f"""
        SELECT DISTINCT ec.ticker, e.event_id, e.event_type, e.event_date_value
        FROM event_companies ec
        JOIN events e ON e.event_id = ec.event_id
        WHERE e.event_type = ?
          AND e.event_date_value BETWEEN ? AND ?
          AND ec.role = ?
          AND ec.ticker IS NOT NULL
    """, [p["E"], p["W_start"], p["W_end"], R]).fetchall()
    return [
        {"firm_ticker": r[0], "event_type": r[2], "anchor_date": fmt_date(r[3]),
         "cited_urls": get_cited_urls(con, [r[1]])}
        for r in rows
    ]


def run_B11(con, p):
    """B.1.1 — events of type E by other firms before/after anchor firm's E event in window."""
    anchor_rows = con.execute("""
        SELECT MIN(event_date_value)
        FROM events
        WHERE event_type = ?
          AND firm_ticker = ?
          AND event_date_value BETWEEN ? AND ?
    """, [p["E"], p["F_anchor"], p["W_start"], p["W_end"]]).fetchone()
    if not anchor_rows or not anchor_rows[0]:
        return []
    anchor_date = anchor_rows[0]
    cmp_op = "<" if p["direction"] == "before" else ">"
    rows = con.execute(f"""
        SELECT firm_ticker, event_type, event_date_value, event_id
        FROM events
        WHERE event_type = ?
          AND event_date_value BETWEEN ? AND ?
          AND event_date_value {cmp_op} ?
          AND firm_ticker IS NOT NULL
          AND firm_ticker != ?
    """, [p["E"], p["W_start"], p["W_end"], anchor_date, p["F_anchor"]]).fetchall()
    return [
        {"firm_ticker": r[0], "event_type": r[1], "anchor_date": fmt_date(r[2]),
         "cited_urls": get_cited_urls(con, [r[3]])}
        for r in rows
    ]


def run_B12(con, p):
    """B.1.2 — firms where E1 AND E2 occurred within |Δ| days (symmetric)."""
    rows = con.execute("""
        SELECT DISTINCT e1.firm_ticker, e1.event_id, e2.event_id
        FROM events e1
        JOIN events e2 ON e1.firm_ticker = e2.firm_ticker
                       AND e1.firm_ticker IS NOT NULL
                       AND e1.event_type = ?
                       AND e2.event_type = ?
                       AND ABS(EPOCH(e1.event_date_value) - EPOCH(e2.event_date_value)) / 86400 <= ?
        WHERE e1.event_date_value BETWEEN ? AND ?
          AND e2.event_date_value BETWEEN ? AND ?
    """, [p["E1"], p["E2"], p["gap_days"], p["W_start"], p["W_end"], p["W_start"], p["W_end"]]).fetchall()
    return [
        {"firm_ticker": r[0], "cited_urls": get_cited_urls(con, [r[1], r[2]])}
        for r in rows
    ]


def run_B13(con, p):
    """B.1.3 — firms where E_trigger followed by E_followup within lag_days."""
    rows = con.execute("""
        SELECT DISTINCT e1.firm_ticker, e1.event_date_value, e2.event_date_value,
               e1.event_id, e2.event_id
        FROM events e1
        JOIN events e2 ON e1.firm_ticker = e2.firm_ticker
                       AND e1.firm_ticker IS NOT NULL
                       AND e1.event_type = ?
                       AND e2.event_type = ?
                       AND (EPOCH(e2.event_date_value) - EPOCH(e1.event_date_value)) / 86400 BETWEEN 0 AND ?
        WHERE e1.event_date_value BETWEEN ? AND ?
          AND e2.event_date_value BETWEEN ? AND ?
    """, [p["E_trigger"], p["E_followup"], p["lag_days"], p["W_start"], p["W_end"], p["W_start"], p["W_end"]]).fetchall()
    return [
        {"firm_ticker": r[0],
         "e_trigger_anchor_date": fmt_date(r[1]),
         "e_followup_anchor_date": fmt_date(r[2]),
         "cited_urls": get_cited_urls(con, [r[3], r[4]])}
        for r in rows
    ]


def run_B15(con, p):
    """B.1.5 — M&A deals announced in W that completed within horizon_days.
    Identity: buyer_ticker, target_ticker, announce_date."""
    rows = con.execute("""
        WITH announces AS (
          SELECT m.event_id, m.cluster_key_hint,
                 m.announcement_date,
                 e.event_date_value as ann_date,
                 buy.ticker as buyer_ticker,
                 tgt.ticker as target_ticker
          FROM events e
          JOIN ma_details m ON m.event_id = e.event_id
          LEFT JOIN event_companies buy ON buy.event_id = e.event_id AND buy.role = 'acquirer'
          LEFT JOIN event_companies tgt ON tgt.event_id = e.event_id AND tgt.role = 'target'
          WHERE e.event_type = 'M&A_announce'
            AND e.event_date_value BETWEEN ? AND ?
        ),
        completes AS (
          SELECT m.event_id, m.cluster_key_hint, e.event_date_value as close_date,
                 buy.ticker as buyer_ticker,
                 tgt.ticker as target_ticker
          FROM events e
          JOIN ma_details m ON m.event_id = e.event_id
          LEFT JOIN event_companies buy ON buy.event_id = e.event_id AND buy.role = 'acquirer'
          LEFT JOIN event_companies tgt ON tgt.event_id = e.event_id AND tgt.role = 'target'
          WHERE e.event_type = 'M&A_complete'
        )
        SELECT a.buyer_ticker, a.target_ticker, a.ann_date, a.event_id, c.event_id
        FROM announces a
        JOIN completes c
          ON (a.cluster_key_hint IS NOT NULL AND a.cluster_key_hint = c.cluster_key_hint)
          OR (a.buyer_ticker IS NOT NULL AND a.buyer_ticker = c.buyer_ticker
              AND a.target_ticker IS NOT NULL AND a.target_ticker = c.target_ticker)
        WHERE (EPOCH(c.close_date) - EPOCH(a.ann_date)) / 86400 BETWEEN 0 AND ?
    """, [p["W_start"], p["W_end"], p["horizon_days"]]).fetchall()
    seen = set()
    out = []
    for r in rows:
        key = (r[0], r[1], fmt_date(r[2]))
        if key in seen or r[0] is None or r[1] is None:
            continue
        seen.add(key)
        out.append({
            "buyer_ticker": r[0],
            "target_ticker": r[1],
            "announce_date": fmt_date(r[2]),
            "cited_urls": get_cited_urls(con, [r[3], r[4]]),
        })
    return out


def run_B16(con, p):
    """B.1.5 (v18 composite_b16) — single-row list with firm_ticker,
    e1_anchor_date, e2_anchor_date, days_between."""
    F = p["F"]
    r1 = con.execute("""
        SELECT MIN(event_date_value) FROM events
        WHERE event_type = ? AND firm_ticker = ? AND event_date_value BETWEEN ? AND ?
    """, [p["E1"], F, p["W_start"], p["W_end"]]).fetchone()
    if not r1 or not r1[0]:
        return []
    d1 = r1[0]
    r2 = con.execute("""
        SELECT MIN(event_date_value) FROM events
        WHERE event_type = ? AND firm_ticker = ? AND event_date_value >= ?
    """, [p["E2"], F, d1]).fetchone()
    if not r2 or not r2[0]:
        return []
    d2 = r2[0]
    return [{
        "firm_ticker": F,
        "e1_anchor_date": fmt_date(d1),
        "e2_anchor_date": fmt_date(d2),
        "days_between": (d2 - d1).days,
    }]


def run_B21(con, p):
    """B.2.1 — ordered by date, position=first/last."""
    order = "ASC" if p["position"] == "first" else "DESC"
    rows = con.execute(f"""
        SELECT firm_ticker, event_type, event_date_value, event_id
        FROM events
        WHERE event_type = ?
          AND event_date_value BETWEEN ? AND ?
          AND firm_ticker IS NOT NULL
        ORDER BY event_date_value {order}, firm_ticker
        LIMIT 1
    """, [p["E"], p["W_start"], p["W_end"]]).fetchall()
    return [
        {"firm_ticker": r[0], "event_type": r[1], "anchor_date": fmt_date(r[2]),
         "cited_urls": get_cited_urls(con, [r[3]])}
        for r in rows
    ]


def run_B22(con, p):
    """B.2.2 — for firm F, list events of types in E_set, chronological."""
    F = p["F"]
    types_placeholders = ",".join(["?"] * len(p["E_set"]))
    rows = con.execute(f"""
        WITH firm_events AS (
          SELECT DISTINCT event_id FROM event_companies WHERE ticker = ?
        )
        SELECT e.firm_ticker, e.event_type, e.event_date_value, e.event_id
        FROM events e
        WHERE (e.firm_ticker = ? OR e.event_id IN (SELECT event_id FROM firm_events))
          AND e.event_type IN ({types_placeholders})
          AND e.event_date_value BETWEEN ? AND ?
        ORDER BY e.event_date_value, e.event_type
    """, [F, F] + list(p["E_set"]) + [p["W_start"], p["W_end"]]).fetchall()
    return [
        {"firm_ticker": r[0] or F, "event_type": r[1], "anchor_date": fmt_date(r[2]),
         "cited_urls": get_cited_urls(con, [r[3]])}
        for r in rows
    ]


def run_B31(con, p):
    """B.3.1 — firms with BOTH E1 AND E2 in window."""
    rows = con.execute("""
        SELECT firm_ticker, MIN(event_id), MAX(event_id) FROM events
        WHERE event_type = ? AND event_date_value BETWEEN ? AND ? AND firm_ticker IS NOT NULL
        GROUP BY firm_ticker
        INTERSECT
        SELECT firm_ticker, MIN(event_id), MAX(event_id) FROM events
        WHERE event_type = ? AND event_date_value BETWEEN ? AND ? AND firm_ticker IS NOT NULL
        GROUP BY firm_ticker
    """, [p["E1"], p["W_start"], p["W_end"], p["E2"], p["W_start"], p["W_end"]]).fetchall()
    # Simpler — INTERSECT on firm_ticker alone, then get cited
    rows = con.execute("""
        SELECT t1.firm_ticker FROM
          (SELECT DISTINCT firm_ticker FROM events
           WHERE event_type = ? AND event_date_value BETWEEN ? AND ? AND firm_ticker IS NOT NULL) t1
        INNER JOIN
          (SELECT DISTINCT firm_ticker FROM events
           WHERE event_type = ? AND event_date_value BETWEEN ? AND ? AND firm_ticker IS NOT NULL) t2
        ON t1.firm_ticker = t2.firm_ticker
    """, [p["E1"], p["W_start"], p["W_end"], p["E2"], p["W_start"], p["W_end"]]).fetchall()
    out = []
    for r in rows:
        ft = r[0]
        evs = [e[0] for e in con.execute(
            "SELECT event_id FROM events WHERE firm_ticker = ? AND event_type IN (?,?) AND event_date_value BETWEEN ? AND ?",
            [ft, p["E1"], p["E2"], p["W_start"], p["W_end"]]).fetchall()]
        out.append({"firm_ticker": ft, "cited_urls": get_cited_urls(con, evs)})
    return out


def run_B32(con, p):
    """B.3.2 — firms with role R in E during BOTH W1 AND W2 windows."""
    R = ROLE_MAP.get(p["R"], p["R"].lower())
    rows = con.execute("""
        SELECT t1.ticker FROM
          (SELECT DISTINCT ec.ticker
           FROM event_companies ec JOIN events e ON e.event_id = ec.event_id
           WHERE ec.role = ? AND e.event_type = ?
             AND e.event_date_value BETWEEN ? AND ? AND ec.ticker IS NOT NULL) t1
        INNER JOIN
          (SELECT DISTINCT ec.ticker
           FROM event_companies ec JOIN events e ON e.event_id = ec.event_id
           WHERE ec.role = ? AND e.event_type = ?
             AND e.event_date_value BETWEEN ? AND ? AND ec.ticker IS NOT NULL) t2
        ON t1.ticker = t2.ticker
    """, [R, p["E"], p["W1_start"], p["W1_end"],
          R, p["E"], p["W2_start"], p["W2_end"]]).fetchall()
    out = []
    for r in rows:
        ft = r[0]
        evs = [e[0] for e in con.execute("""
            SELECT DISTINCT ec.event_id
            FROM event_companies ec JOIN events e ON e.event_id = ec.event_id
            WHERE ec.role = ? AND e.event_type = ? AND ec.ticker = ?
              AND ((e.event_date_value BETWEEN ? AND ?) OR (e.event_date_value BETWEEN ? AND ?))
        """, [R, p["E"], ft, p["W1_start"], p["W1_end"], p["W2_start"], p["W2_end"]]).fetchall()]
        out.append({"firm_ticker": ft, "cited_urls": get_cited_urls(con, evs)})
    return out


def run_B33(con, p):
    """B.3.3 — firms with role R in BOTH E1 AND E2 events in window."""
    R = ROLE_MAP.get(p["R"], p["R"].lower())
    rows = con.execute("""
        SELECT t1.ticker FROM
          (SELECT DISTINCT ec.ticker
           FROM event_companies ec JOIN events e ON e.event_id = ec.event_id
           WHERE ec.role = ? AND e.event_type = ?
             AND e.event_date_value BETWEEN ? AND ? AND ec.ticker IS NOT NULL) t1
        INNER JOIN
          (SELECT DISTINCT ec.ticker
           FROM event_companies ec JOIN events e ON e.event_id = ec.event_id
           WHERE ec.role = ? AND e.event_type = ?
             AND e.event_date_value BETWEEN ? AND ? AND ec.ticker IS NOT NULL) t2
        ON t1.ticker = t2.ticker
    """, [R, p["E1"], p["W_start"], p["W_end"],
          R, p["E2"], p["W_start"], p["W_end"]]).fetchall()
    out = []
    for r in rows:
        ft = r[0]
        evs = [e[0] for e in con.execute("""
            SELECT DISTINCT ec.event_id
            FROM event_companies ec JOIN events e ON e.event_id = ec.event_id
            WHERE ec.role = ? AND e.event_type IN (?, ?) AND ec.ticker = ?
              AND e.event_date_value BETWEEN ? AND ?
        """, [R, p["E1"], p["E2"], ft, p["W_start"], p["W_end"]]).fetchall()]
        out.append({"firm_ticker": ft, "cited_urls": get_cited_urls(con, evs)})
    return out


def run_B41(con, p):
    """B.4.1 — firms with ≥N events of type E in window."""
    rows = con.execute("""
        SELECT firm_ticker, COUNT(*) FROM events
        WHERE event_type = ? AND event_date_value BETWEEN ? AND ?
          AND firm_ticker IS NOT NULL
        GROUP BY firm_ticker HAVING COUNT(*) >= ?
    """, [p["E"], p["W_start"], p["W_end"], p["N"]]).fetchall()
    out = []
    for r in rows:
        ft = r[0]
        evs = [e[0] for e in con.execute("""
            SELECT event_id FROM events
            WHERE event_type = ? AND firm_ticker = ?
              AND event_date_value BETWEEN ? AND ?
        """, [p["E"], ft, p["W_start"], p["W_end"]]).fetchall()]
        out.append({"firm_ticker": ft, "cited_urls": get_cited_urls(con, evs)})
    return out


def run_B42(con, p):
    """B.4.2 — firms playing role R in ≥N events of type E in window."""
    R = ROLE_MAP.get(p["R"], p["R"].lower())
    rows = con.execute("""
        SELECT ec.ticker, COUNT(DISTINCT e.event_id)
        FROM event_companies ec JOIN events e ON e.event_id = ec.event_id
        WHERE ec.role = ? AND e.event_type = ?
          AND e.event_date_value BETWEEN ? AND ? AND ec.ticker IS NOT NULL
        GROUP BY ec.ticker HAVING COUNT(DISTINCT e.event_id) >= ?
    """, [R, p["E"], p["W_start"], p["W_end"], p["N"]]).fetchall()
    out = []
    for r in rows:
        ft = r[0]
        evs = [e[0] for e in con.execute("""
            SELECT DISTINCT ec.event_id
            FROM event_companies ec JOIN events e ON e.event_id = ec.event_id
            WHERE ec.role = ? AND e.event_type = ? AND ec.ticker = ?
              AND e.event_date_value BETWEEN ? AND ?
        """, [R, p["E"], ft, p["W_start"], p["W_end"]]).fetchall()]
        out.append({"firm_ticker": ft, "cited_urls": get_cited_urls(con, evs)})
    return out


def run_B43(con, p):
    """B.4.3 — list events of type E in window, attach time bucket."""
    bucket_unit = p["bucket_unit"]  # week | month | quarter
    rows = con.execute("""
        SELECT firm_ticker, event_type, event_date_value, event_id
        FROM events
        WHERE event_type = ? AND event_date_value BETWEEN ? AND ?
          AND firm_ticker IS NOT NULL
        ORDER BY event_date_value
    """, [p["E"], p["W_start"], p["W_end"]]).fetchall()

    def bucket_label(d):
        if d is None:
            return None
        if bucket_unit == "week":
            iso = d.isocalendar()
            return f"{iso.year}-W{iso.week:02d}"
        if bucket_unit == "month":
            return d.strftime("%Y-%m")
        if bucket_unit == "quarter":
            q = (d.month - 1) // 3 + 1
            return f"{d.year}-Q{q}"
        return d.strftime("%Y-%m-%d")

    return [
        {"firm_ticker": r[0], "event_type": r[1], "anchor_date": fmt_date(r[2]),
         "bucket": bucket_label(r[2]),
         "cited_urls": get_cited_urls(con, [r[3]])}
        for r in rows
    ]


def run_B44(con, p):
    """B.4.4 — list events of types in E_set in window, with event_type label."""
    types_placeholders = ",".join(["?"] * len(p["E_set"]))
    rows = con.execute(f"""
        SELECT firm_ticker, event_type, event_date_value, event_id
        FROM events
        WHERE event_type IN ({types_placeholders})
          AND event_date_value BETWEEN ? AND ?
          AND firm_ticker IS NOT NULL
    """, list(p["E_set"]) + [p["W_start"], p["W_end"]]).fetchall()
    return [
        {"firm_ticker": r[0], "event_type": r[1], "anchor_date": fmt_date(r[2]),
         "cited_urls": get_cited_urls(con, [r[3]])}
        for r in rows
    ]


# ────────────────────────────────────────────────────────────────────────────
# Dispatch
# ────────────────────────────────────────────────────────────────────────────

DISPATCH = {
    "A.1.1": run_A11,
    "A.1.2": run_A12,
    "A.2.1": run_A21,
    "A.3.1": run_A31,
    "B.1.1": run_B11,
    "B.1.2": run_B12,
    "B.1.3": run_B13,
    "B.1.4": run_B15,        # paper B.1.4 (announce->complete identity)
    "B.1.5": run_B16,        # paper B.1.5 (date arithmetic on self-join)
    "B.2.1": run_B21,
    "B.2.2": run_B22,
    "B.3.1": run_B31,
    "B.3.2": run_B32,
    "B.3.3": run_B33,
    "B.4.1": run_B41,
    "B.4.2": run_B42,
    "B.4.3": run_B43,
    "B.4.4": run_B44,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    con = duckdb.connect(args.db, read_only=True)
    print(f"DB opened: {args.db}")

    # Load questions
    open_fn = gzip.open if args.questions.endswith(".gz") else open
    with open_fn(args.questions, "rt") as f:
        questions = [json.loads(l) for l in f if l.strip()]
    print(f"Questions: {len(questions)}")

    n_dispatched = 0
    n_unsupported = 0
    n_error = 0
    template_counts = defaultdict(lambda: {"ok": 0, "err": 0, "skip": 0})

    with open(args.output, "w") as fout:
        for q in questions:
            tid = q["id"]
            qid = q["qid"]
            output_id = tid
            try:
                if tid == "A.3.combined":
                    answer = run_A3combined(con, q["params"], q)
                    # Map A.3.combined → A.3.1 (entities, T18) or A.3.3 (events, T40)
                    # to match eval.py templates_config.json entries
                    legacy = q.get("legacy_id", "")
                    output_id = "A.3.1" if legacy == "T18" else "A.3.3"
                elif tid in DISPATCH and DISPATCH[tid] is not None:
                    answer = DISPATCH[tid](con, q["params"])
                else:
                    n_unsupported += 1
                    template_counts[tid]["skip"] += 1
                    continue
                template_counts[tid]["ok"] += 1
                n_dispatched += 1
            except Exception as e:
                n_error += 1
                template_counts[tid]["err"] += 1
                answer = []
                print(f"  ERR {tid} {qid}: {type(e).__name__}: {str(e)[:200]}")

            fout.write(json.dumps({"qid": qid, "id": output_id, "answer": answer}) + "\n")

    print(f"\nDispatched: {n_dispatched}  Errors: {n_error}  Unsupported: {n_unsupported}")
    print(f"\nPer-template:")
    for tid in sorted(template_counts):
        s = template_counts[tid]
        print(f"  {tid:<14}: ok={s['ok']:>3}  err={s['err']:>3}  skip={s['skip']:>3}")
    print(f"\nPredictions written: {args.output}")


if __name__ == "__main__":
    main()
