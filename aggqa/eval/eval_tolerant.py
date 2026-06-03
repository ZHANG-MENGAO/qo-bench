#!/usr/bin/env python3
# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Patched tolerant scorer (eval_tolerant_v2; works with v3 bundle).

Source: eval_bundle_v3_2026-05-21/eval_tolerant.py (byte-identical to v2 5/19).
Patches applied 2026-05-20 to mirror eval_v2.py audit fixes that materially
affect cap50 scoring:
  - Fix #2: per-question judge_status filter (A.2.1 50q gemma-only)
  - Fix #1: preserves_order positional matching (B.2.2 42q)
  - Fix #3: multi-format date parse (YYYY/MM/DD, YYYYMMDD, ...)
  - Fix #4: _to_int coercion for B.1.5 days_between
  - Fix #6: prefix scan for golden_chunks* (defense-in-depth)
  - Fix #7: qualifying_events nested traversal (defense-in-depth)

ORIGINAL DOCSTRING BELOW.
Tolerant variant of eval.py — relaxes date matching by ±TOLERANCE_DAYS.

Why: The strict eval.py requires anchor_date / *_date fields to match exactly.
FNSPID news articles often report the event's effective date (e.g. "effective
Jul 1, 2022"), while KeyDev GT uses the announce/report date (e.g. 2022-07-06).
Both are legitimate "event dates" but differ by 1-5 days, so strict matching
forfeits recall even when the firm and event_type are correct.

Per design decision (2026-05-14): tolerance is 7 days.

Same CLI as eval.py. Run side-by-side:
  python eval.py           ... --output results_strict.json
  python eval_tolerant.py  ... --output results_tolerant.json

Differences from eval.py:
  - For `list_recall` templates, identity-key matching tolerates a ±7-day
    difference on any field whose name is "date" or ends with "_date".
  - Non-date identity_keys (firm_ticker, event_type, buyer_ticker, etc.)
    still match exactly.
  - Provenance (with_prov) scoring is unchanged — cited_urls must still
    exactly intersect golden_chunks URLs.
  - `int_match` and `empty_check` scorers are unchanged (no dates involved).
"""

import argparse
import datetime
import gzip
import json
import sys
from collections import defaultdict
from itertools import permutations


TOLERANCE_DAYS = 7


# ---------- IO (verbatim from eval.py) ----------

def open_maybe_gzip(path):
    return (gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz')
            else open(path, 'r', encoding='utf-8'))


def load_questions(path):
    """Accept JSONL (.jsonl[.gz]) or JSON document with .questions[]."""
    qs = {}
    is_jsonl = path.endswith('.jsonl') or path.endswith('.jsonl.gz')
    with open_maybe_gzip(path) as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line: continue
                q = json.loads(line)
                qs[q['qid']] = q
        else:
            d = json.load(f)
            for q in d.get('questions', []):
                qs[q['qid']] = q
    return qs


def load_predictions(path):
    preds = {}
    with open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            preds[p['qid']] = p
    return preds


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------- Date helpers ----------

def _parse_date(s):
    """Fix #3 (audit 2026-05-19): accept multiple date formats, not just ISO.
    Returns datetime.date or None on failure."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y%m%d'):
        try:
            return datetime.datetime.strptime(s[:10] if fmt != '%Y%m%d' else s[:8], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _to_int(v):
    """Fix #4 (audit 2026-05-19): coerce numeric-string to int. None on failure."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v.strip())
        except (ValueError, TypeError):
            return None
    return None


def _b16_extract_dates(gt):
    """Extract e1/e2 anchor dates from a v3 B.1.5 GT row.

    v3 GT shape stores anchor dates nested inside qualifying_events:
      qualifying_events: [{role: "e1", anchor_date: "..."}, {role: "e2", ...}]
    v2 GT shape (and the synthetic null-anchor test) stores them flat:
      {e1_anchor_date: "...", e2_anchor_date: "..."}
    Try the nested v3 path first; fall back to flat v2 path."""
    e1, e2 = gt.get('e1_anchor_date'), gt.get('e2_anchor_date')
    for ev in gt.get('qualifying_events') or []:
        if not isinstance(ev, dict):
            continue
        role = ev.get('role')
        if role == 'e1' and e1 is None:
            e1 = ev.get('anchor_date')
        elif role == 'e2' and e2 is None:
            e2 = ev.get('anchor_date')
    return e1, e2


def _date_within_tol(gv, pv, tol_days):
    """True if both gv and pv parse to dates within tol_days of each other.
    Treats both-None as a match (same as exact eval); one-None as no match."""
    g = _parse_date(gv) if gv else None
    p = _parse_date(pv) if pv else None
    if g is None and p is None:
        return gv == pv  # match if both are exactly equal (including both None)
    if g is None or p is None:
        return False
    return abs((g - p).days) <= tol_days


def _is_date_field(name):
    """Field-name heuristic: anything named "date" or ending in "_date"."""
    return name == 'date' or name.endswith('_date')


def _bipartite_match_within_tol(gt_dates, pred_dates, tol):
    """Try to match each gt_date to a unique pred_date within tol days.
    Brute-force permutations (N ≤ 4 in all current schemas)."""
    if len(gt_dates) != len(pred_dates):
        return False
    if not gt_dates:
        return True
    for perm in permutations(pred_dates):
        if all(_date_within_tol(g, p, tol) for g, p in zip(gt_dates, perm)):
            return True
    return False


# ---------- Identity matching with tolerance ----------

def _matches_with_tolerance(gt_row, pred_row, identity_keys,
                            identity_keys_sorted, tol):
    """True if pred_row matches gt_row under tolerant identity rules."""
    if not isinstance(pred_row, dict):
        return False
    # Positional identity keys: exact for non-date fields, tolerant for date.
    for k in identity_keys or []:
        gv = gt_row.get(k)
        pv = pred_row.get(k)
        if _is_date_field(k):
            if not _date_within_tol(gv, pv, tol):
                return False
        else:
            if gv != pv:
                return False
    # Sorted (order-independent) keys.
    sorted_keys = identity_keys_sorted or []
    if sorted_keys:
        all_dates = all(_is_date_field(k) for k in sorted_keys)
        if all_dates:
            gv = [gt_row.get(k) for k in sorted_keys]
            pv = [pred_row.get(k) for k in sorted_keys]
            if not _bipartite_match_within_tol(gv, pv, tol):
                return False
        else:
            # Mixed/non-date sorted keys: fall back to sorted-tuple equality
            # (same as strict eval.py).
            g_sorted = tuple(sorted([gt_row.get(k) for k in sorted_keys],
                                    key=lambda x: (x is None, x)))
            p_sorted = tuple(sorted([pred_row.get(k) for k in sorted_keys],
                                    key=lambda x: (x is None, x)))
            if g_sorted != p_sorted:
                return False
    return True


# ---------- Scoring helpers (golden chunk URLs unchanged) ----------

def collect_gt_urls(gt_row):
    """Fix #6 + #7 (audit 2026-05-19): prefix scan for any golden_chunks*
    field + traversal of qualifying_events[*].golden_chunks (per-event
    grounding added in eval_bundle v7)."""
    urls = set()
    if not isinstance(gt_row, dict):
        return urls
    for k, v in gt_row.items():
        if not k.startswith('golden_chunks'):
            continue
        if not isinstance(v, list):
            continue
        for chunk in v:
            u = chunk.get('article_url') if isinstance(chunk, dict) else None
            if u:
                urls.add(u)
    for ev in gt_row.get('qualifying_events') or []:
        if not isinstance(ev, dict):
            continue
        for chunk in ev.get('golden_chunks') or []:
            u = chunk.get('article_url') if isinstance(chunk, dict) else None
            if u:
                urls.add(u)
    return urls


def question_supports_provenance(record, cfg):
    """Fix #2 (audit 2026-05-19): exclude posthoc_gemma_only attestations from
    with_prov scoring (single-judge GT is insufficient evidence for grounding)."""
    if not cfg.get('supports_provenance'):
        return False
    judge = (record.get('metadata') or {}).get('judge_status')
    if judge == 'posthoc_gemma_only':
        return False
    return True


# ---------- Scorers ----------

def score_int(record, prediction, cfg):
    """Unchanged from eval.py — no dates involved."""
    field = cfg['scalar_field']
    gt_row = record['gt'][0] if record['gt'] else None
    gt_int = gt_row.get(field) if gt_row else None
    pred_ans = prediction.get('answer')
    match = (isinstance(pred_ans, (int, float))
             and gt_int is not None
             and int(pred_ans) == int(gt_int))
    return {'recall_no_prov': 1.0 if match else 0.0,
            'recall_with_prov': None}


def score_empty(record, prediction, cfg):
    """Unchanged from eval.py — no dates involved."""
    pred_ans = prediction.get('answer')
    empty = (pred_ans is None
             or (isinstance(pred_ans, list) and len(pred_ans) == 0))
    return {'recall_no_prov': 1.0 if empty else 0.0,
            'recall_with_prov': None}


def score_list_recall(record, prediction, cfg):
    """Tolerant version: for each GT row, find ANY pred whose identity keys
    match within ±TOLERANCE_DAYS on date fields and exactly on non-date fields.

    Fix #1: when preserves_order=true, match positionally (B.2.2).
    Fix #2: gemma-only attestations excluded from with_prov denominator."""
    identity_keys = cfg.get('identity_keys') or []
    identity_keys_sorted = cfg.get('identity_keys_sorted') or []
    gt_rows = record.get('gt') or []
    pred_items = prediction.get('answer') or []
    if not isinstance(pred_items, list):
        pred_items = []
    supports_prov_local = question_supports_provenance(record, cfg)

    if not gt_rows:
        empty = (len(pred_items) == 0)
        wp = (None if not supports_prov_local else (1.0 if empty else 0.0))
        return {'recall_no_prov': 1.0 if empty else 0.0, 'recall_with_prov': wp}

    n_total = len(gt_rows)

    # Fix #1: positional matching for ordered templates (B.2.2).
    if cfg.get('preserves_order'):
        matched_no_prov = 0
        matched_with_prov = 0
        for i, gt_row in enumerate(gt_rows):
            if i >= len(pred_items):
                break
            pr = pred_items[i]
            if not _matches_with_tolerance(gt_row, pr, identity_keys,
                                           identity_keys_sorted, TOLERANCE_DAYS):
                continue
            matched_no_prov += 1
            if supports_prov_local:
                gt_urls = collect_gt_urls(gt_row)
                cited = set((pr or {}).get('cited_urls') or []) if isinstance(pr, dict) else set()
                if cited & gt_urls:
                    matched_with_prov += 1
        return {
            'recall_no_prov':   matched_no_prov / n_total,
            'recall_with_prov': (matched_with_prov / n_total) if supports_prov_local else None,
        }

    matched_no_prov = 0
    matched_with_prov = 0

    for gt_row in gt_rows:
        matching_preds = [
            pr for pr in pred_items
            if _matches_with_tolerance(gt_row, pr, identity_keys,
                                       identity_keys_sorted, TOLERANCE_DAYS)
        ]
        if matching_preds:
            matched_no_prov += 1
            if supports_prov_local:
                gt_urls = collect_gt_urls(gt_row)
                grounded = any(
                    bool(set(pr.get('cited_urls') or []) & gt_urls)
                    for pr in matching_preds
                )
                if grounded:
                    matched_with_prov += 1

    return {
        'recall_no_prov':   matched_no_prov / n_total,
        'recall_with_prov': (matched_with_prov / n_total) if supports_prov_local else None,
    }




INT_TOLERANCE_DAYS = 14  # +/-14 for days_between (= 2 * TOLERANCE_DAYS, since each endpoint can be off +/-TOLERANCE_DAYS)

def score_composite_b16(record, prediction, cfg):
    """B.1.5 tolerant composite: dates within +/-TOLERANCE_DAYS; days_between within +/-INT_TOLERANCE_DAYS.

    v3 migration: GT rows store anchor dates inside qualifying_events[*].anchor_date
    rather than as flat top-level e1_anchor_date / e2_anchor_date fields. Use
    _b16_extract_dates() to handle both shapes transparently (mirrors eval_v2.py)."""
    gt = (record.get('gt') or [None])[0]
    if not gt:
        return {'recall_no_prov': 0.0, 'recall_with_prov': None}
    pred_ans = prediction.get('answer') or []
    if not isinstance(pred_ans, list) or not pred_ans:
        return {'recall_no_prov': 0.0, 'recall_with_prov': None}
    p = pred_ans[0] if isinstance(pred_ans[0], dict) else {}
    raw_e1, raw_e2 = _b16_extract_dates(gt)
    d1_ok = _date_within_tol(raw_e1, p.get('e1_anchor_date'), TOLERANCE_DAYS)
    d2_ok = _date_within_tol(raw_e2, p.get('e2_anchor_date'), TOLERANCE_DAYS)
    # Fix #4: coerce numeric-string to int.
    db_p = _to_int(p.get('days_between'))
    db_g = _to_int(gt.get('days_between'))
    db_ok = (db_p is not None and db_g is not None
             and abs(db_p - db_g) <= INT_TOLERANCE_DAYS)
    return {'recall_no_prov': 1.0 if (d1_ok and d2_ok and db_ok) else 0.0,
            'recall_with_prov': None}

SCORERS = {
    'int_match':     score_int,
    'empty_check':   score_empty,
    'list_recall':   score_list_recall,
    'composite_b16': score_composite_b16,
}


def score_one(record, prediction, cfg):
    return SCORERS[cfg['scoring']](record, prediction, cfg)


# ---------- Aggregation (verbatim from eval.py) ----------

def aggregate(per_q):
    by_id        = defaultdict(lambda: {'n': 0, 'np': 0.0, 'wp': 0.0, 'wp_n': 0})
    by_sub_axis  = defaultdict(lambda: {'n': 0, 'np': 0.0, 'wp': 0.0, 'wp_n': 0})
    by_cap       = defaultdict(lambda: {'n': 0, 'np': 0.0, 'wp': 0.0, 'wp_n': 0})
    overall      = {'n': 0, 'np': 0.0, 'wp': 0.0, 'wp_n': 0}

    for s in per_q:
        np = s['recall_no_prov']
        wp = s.get('recall_with_prov')
        for bucket in (by_id[s['id']],
                       by_sub_axis[s['sub_axis']],
                       by_cap[s['cap']],
                       overall):
            bucket['n']  += 1
            bucket['np'] += np
            if wp is not None:
                bucket['wp']   += wp
                bucket['wp_n'] += 1

    def finalize(b):
        out = {'n': b['n'],
               'mean_recall_no_prov': (b['np'] / b['n']) if b['n'] else None}
        if b['wp_n']:
            out['mean_recall_with_prov'] = b['wp'] / b['wp_n']
            out['n_with_provenance']     = b['wp_n']
        return out

    return {
        'by_id':       {k: finalize(v) for k, v in by_id.items()},
        'by_sub_axis': {k: finalize(v) for k, v in by_sub_axis.items()},
        'by_cap':      {k: finalize(v) for k, v in by_cap.items()},
        'overall':     finalize(overall),
    }


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description=f'Aggregate-QA scorer with ±{TOLERANCE_DAYS}-day date tolerance.'
    )
    ap.add_argument('--questions',   required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--config',      required=True)
    ap.add_argument('--output',      required=True)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    preds     = load_predictions(args.predictions)
    config    = load_config(args.config)

    per_q = []
    missing_pred = []
    unknown_id   = []
    for qid, record in questions.items():
        cfg = config.get(record['id'])
        if cfg is None:
            unknown_id.append(qid)
            continue
        pred = preds.get(qid)
        if pred is None:
            missing_pred.append(qid)
            per_q.append({
                'qid':              qid,
                'id':               record['id'],
                'cap':              record['cap'],
                'sub_axis':         record['sub_axis'],
                'recall_no_prov':   0.0,
                'recall_with_prov': None,
            })
            continue
        scores = score_one(record, pred, cfg)
        per_q.append({
            'qid':              qid,
            'id':               record['id'],
            'cap':              record['cap'],
            'sub_axis':         record['sub_axis'],
            **scores,
        })

    summary = aggregate(per_q)
    summary['tolerance_days']   = TOLERANCE_DAYS
    summary['n_questions']      = len(questions)
    summary['n_predictions']    = len(preds)
    summary['n_missing_preds']  = len(missing_pred)
    summary['n_unknown_ids']    = len(unknown_id)
    if missing_pred[:10]:
        summary['sample_missing_qids'] = missing_pred[:10]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    o = summary['overall']
    print(f"[tolerant ±{TOLERANCE_DAYS}d] "
          f"questions={summary['n_questions']}  "
          f"predictions={summary['n_predictions']}  "
          f"missing={summary['n_missing_preds']}")
    print(f"overall mean recall (no_prov):   "
          f"{o.get('mean_recall_no_prov', 0):.3f}")
    if 'mean_recall_with_prov' in o:
        print(f"overall mean recall (with_prov): "
              f"{o['mean_recall_with_prov']:.3f} "
              f"(over {o['n_with_provenance']} provenance-applicable questions)")


if __name__ == '__main__':
    main()
