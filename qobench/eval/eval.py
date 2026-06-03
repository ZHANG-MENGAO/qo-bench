#!/usr/bin/env python3
# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-NC-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Patched copy of eval_bundle_v3_2026-05-21/eval.py (byte-identical to v2's).

This file is NOT an original scorer — it is the upstream `eval.py` with six
targeted bug fixes applied (issues #1, #2, #3, #4, #6, #7 from the 2026-05-19
audit). Use this file (not the bundle eval.py) for all QO-Bench paper
numbers. The v3 bundle's eval.py has Fixes #6 (golden_chunks prefix scan) and
#7 (qualifying_events nested traversal) inlined, but Fixes #1-4 are still
missing upstream and matter for correctness:

  #1: B.2.2 preserves_order=true → positional matching (set match → false +)
  #2: per-question judge_status filter (no-op on v3 since all chunks are 3-of-3)
  #3: multi-format date parse (YYYY-MM-DD / YYYY/MM/DD / YYYYMMDD)
  #4: _to_int coercion for B.1.5 days_between (LLM may emit string "30")

The 4 fixes are documented in the source code below with inline `Fix #N:` comments.

Four scoring modes, dispatched by `templates_config.json[id].scoring`:
  - list_recall:   set-intersection recall on identity keys.
  - int_match:     exact integer match against the GT scalar.
  - empty_check:   prediction must be an empty list.
  - composite_b16: B.1.5 scalar tuple match (firm + e1 + e2 + days_between).

Two scores reported per question (for list_recall templates):
  - recall_no_prov:   identity-key match only.
  - recall_with_prov: identity-key match AND at least one of the prediction's
                      `cited_urls` appears in the GT row's `golden_chunks`.

For int_match / empty_check / composite_b16 templates, only recall_no_prov is
reported. recall_with_prov is also None for `list_recall` questions whose
metadata flags `judge_status == "posthoc_gemma_only"` (single-judge attestation
is insufficient evidence for provenance scoring — see audit issue #2).

Predictions file (JSONL, one record per line):
    {"qid": "<uuid>", "answer": <typed answer per output_schema>}

Usage (the paper's canonical eval target: 785 questions):
    python3 -m qobench.eval.eval_tolerant \\
        --questions    benchmark/questions/questions.jsonl.gz \\
        --predictions  predictions.jsonl \\
        --config       benchmark/templates_config.json \\
        --output       results.json
"""

import argparse
import datetime
import gzip
import json
import sys
from collections import defaultdict


# ---------- IO ----------

def open_maybe_gzip(path):
    return (gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz')
            else open(path, 'r', encoding='utf-8'))


def load_questions(path):
    """Accept either JSONL (one record per line, files ending in .jsonl[.gz])
    or a single JSON document with `.questions[]` (e.g. sample_questions.json)."""
    qs = {}
    is_jsonl = path.endswith('.jsonl') or path.endswith('.jsonl.gz')
    with open_maybe_gzip(path) as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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


# ---------- Scoring helpers ----------

def collect_gt_urls(gt_row):
    """Pool all article URLs from every golden_chunks* field on a GT row,
    plus chunks nested inside qualifying_events[*].golden_chunks.

    Fix #6 (audit 2026-05-19): prefix scan rather than hard-coded suffix list.
    The pre-fix code enumerated 8 known suffixes; any new suffix added upstream
    would be silently dropped, causing with_prov to under-report. Now any key
    starting with 'golden_chunks' is pooled.

    Fix #7 (upstream v7 2026-05-19 per-event grounding): B.3.1/2/3 + B.4.1/2 now
    carry `qualifying_events: [{golden_chunks: [...]}, ...]` per GT row instead
    of flat top-level chunks. Mirror the upstream bundle eval.py / eval_tolerant.py
    update — without nested traversal, with_prov scoring on those templates
    silently under-reports. B.1.1's question-level anchor_event is attached as
    row-level `golden_chunks_anchor` by the caller, covered by Fix #6's prefix
    scan."""
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


def _norm_date(v):
    """Fix #3 (audit 2026-05-19): normalize date strings for identity-key
    matching. Returns ISO-8601 (YYYY-MM-DD) if v parses as a date in any of
    the common formats; returns the original value otherwise (so mismatch
    still surfaces as 0 recall, no silent acceptance). Strict semantics
    preserved — no date tolerance, only formatting variation."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return v
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y%m%d'):
        try:
            return datetime.datetime.strptime(s[:10], fmt).date().isoformat()
        except (ValueError, TypeError):
            continue
    return v


def make_key_fn(cfg):
    """Build a function (record) -> hashable identity key based on cfg.
    Date-valued identity_keys get normalized via _norm_date (fix #3)."""
    keys = cfg.get('identity_keys') or []
    sorted_keys = cfg.get('identity_keys_sorted') or []

    def _val(r, k):
        v = r.get(k)
        if k == 'date' or k.endswith('_date'):
            v = _norm_date(v)
        return v

    def key(r):
        if not isinstance(r, dict):
            return None
        base = tuple(_val(r, k) for k in keys)
        if sorted_keys:
            sk = tuple(sorted([_val(r, k) for k in sorted_keys],
                              key=lambda x: (x is None, x)))
            base = base + (sk,)
        return base
    return key


# ---------- Scorers ----------

def question_supports_provenance(record, cfg):
    """Fix #2 (audit 2026-05-19): per-question (not just per-template) prov
    applicability. Templates with supports_provenance=true can still contain
    questions whose GT attestation is single-judge (gemma-only), insufficient
    for grounding scoring. Returns False for those; True otherwise (if the
    template supports prov)."""
    if not cfg.get('supports_provenance'):
        return False
    judge = (record.get('metadata') or {}).get('judge_status')
    if judge == 'posthoc_gemma_only':
        return False
    return True


def score_int(record, prediction, cfg):
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
    pred_ans = prediction.get('answer')
    empty = (pred_ans is None
             or (isinstance(pred_ans, list) and len(pred_ans) == 0))
    return {'recall_no_prov': 1.0 if empty else 0.0,
            'recall_with_prov': None}


def score_list_recall(record, prediction, cfg):
    key = make_key_fn(cfg)
    gt_rows = record.get('gt') or []
    pred_items = prediction.get('answer') or []
    if not isinstance(pred_items, list):
        pred_items = []

    gt_keys = [key(r) for r in gt_rows]
    gt_keyset = set(gt_keys)

    pred_keys = [key(r) for r in pred_items]
    # Index pred items by key (may be multiple if model returns duplicates)
    pred_by_key = defaultdict(list)
    for r, k in zip(pred_items, pred_keys):
        if k is not None:
            pred_by_key[k].append(r)

    if not gt_keys:
        # Edge case: GT empty (shouldn't happen for list_recall but handle it)
        empty = (len(pred_items) == 0)
        return {'recall_no_prov': 1.0 if empty else 0.0,
                'recall_with_prov': None if not cfg.get('supports_provenance')
                                   else (1.0 if empty else 0.0)}

    n_total = len(gt_keys)

    # Fix #1 (audit 2026-05-19): when preserves_order=true, match positionally
    # rather than as a set. Otherwise B.2.2 ('chronologically ascending') is
    # silently set-matched and a model that returns events in arbitrary order
    # still scores 1.0.
    if cfg.get('preserves_order'):
        supports_prov_local = question_supports_provenance(record, cfg)
        matched_no_prov = 0
        matched_with_prov = 0
        for i, (gk, gt_row) in enumerate(zip(gt_keys, gt_rows)):
            if i >= len(pred_keys):
                break
            if pred_keys[i] != gk:
                continue
            matched_no_prov += 1
            if supports_prov_local:
                gt_urls = collect_gt_urls(gt_row)
                pred_item = pred_items[i] if isinstance(pred_items[i], dict) else {}
                cited = set(pred_item.get('cited_urls') or [])
                if cited & gt_urls:
                    matched_with_prov += 1
        return {
            'recall_no_prov':    matched_no_prov / n_total,
            'recall_with_prov':  (matched_with_prov / n_total) if supports_prov_local else None,
        }

    matched_no_prov = 0
    matched_with_prov = 0

    supports_prov = question_supports_provenance(record, cfg)

    for gt_row, gk in zip(gt_rows, gt_keys):
        if gk in pred_by_key:
            matched_no_prov += 1
            if supports_prov:
                gt_urls = collect_gt_urls(gt_row)
                # any pred item with this key has at least one matching URL?
                grounded = False
                for pr in pred_by_key[gk]:
                    cited = set(pr.get('cited_urls') or [])
                    if cited & gt_urls:
                        grounded = True
                        break
                if grounded:
                    matched_with_prov += 1

    return {
        'recall_no_prov':    matched_no_prov / n_total,
        'recall_with_prov':  (matched_with_prov / n_total) if supports_prov else None,
    }




def _to_int(v):
    """Fix #4 (audit 2026-05-19): coerce numeric-string to int. None on failure.
    Rejects bool (which Python treats as int subclass) since 'True'/'False'
    are not numeric in this context."""
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


def score_composite_b16(record, prediction, cfg):
    """B.1.5 strict composite: both anchor dates AND days_between must match.

    Fixes #3 + #4 (audit 2026-05-19): anchor dates normalized via _norm_date;
    days_between coerced via _to_int. Both sides (pred + GT) normalized so
    comparison is structural, not stringly-typed.

    v3 migration: GT rows store anchor dates inside qualifying_events[*].anchor_date
    rather than as flat top-level e1_anchor_date / e2_anchor_date fields. Use
    _b16_extract_dates() to handle both shapes transparently."""
    gt = (record.get('gt') or [None])[0]
    if not gt:
        return {'recall_no_prov': 0.0, 'recall_with_prov': None}
    pred_ans = prediction.get('answer') or []
    if not isinstance(pred_ans, list) or not pred_ans:
        return {'recall_no_prov': 0.0, 'recall_with_prov': None}
    p = pred_ans[0] if isinstance(pred_ans[0], dict) else {}

    raw_e1, raw_e2 = _b16_extract_dates(gt)
    g_e1 = _norm_date(raw_e1)
    g_e2 = _norm_date(raw_e2)
    p_e1 = _norm_date(p.get('e1_anchor_date'))
    p_e2 = _norm_date(p.get('e2_anchor_date'))

    d1_ok = bool(g_e1) and p_e1 == g_e1
    d2_ok = bool(g_e2) and p_e2 == g_e2

    db_p = _to_int(p.get('days_between'))
    db_g = _to_int(gt.get('days_between'))
    db_ok = (db_p is not None and db_g is not None and db_p == db_g)

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


# ---------- Aggregation ----------

def aggregate(per_q):
    """Aggregate per-question scores into by_id / by_sub_axis / by_cap /
    overall.

    `per_q` is a list of dicts with keys: id, cap, sub_axis,
    recall_no_prov, recall_with_prov (or None).
    """
    by_id        = defaultdict(lambda: {'n': 0, 'np': 0.0, 'wp': 0.0,
                                         'wp_n': 0})
    by_sub_axis  = defaultdict(lambda: {'n': 0, 'np': 0.0, 'wp': 0.0,
                                         'wp_n': 0})
    by_cap       = defaultdict(lambda: {'n': 0, 'np': 0.0, 'wp': 0.0,
                                         'wp_n': 0})
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
    ap = argparse.ArgumentParser()
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
            # No prediction = 0 recall
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
    summary['n_questions']      = len(questions)
    summary['n_predictions']    = len(preds)
    summary['n_missing_preds']  = len(missing_pred)
    summary['n_unknown_ids']    = len(unknown_id)
    if missing_pred[:10]:
        summary['sample_missing_qids'] = missing_pred[:10]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    # Brief stdout summary
    o = summary['overall']
    print(f"questions={summary['n_questions']}  "
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
