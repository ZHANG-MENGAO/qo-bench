#!/usr/bin/env python3
# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Evaluation script for the v2 Aggregate-QA naive-RAG pilot benchmark.

Deterministic scoring: tuple set intersection with date tolerance, no LLM
judge. Per v2 spec §3.5 / §3 — recall on the matched tuple set is the
primary metric. T26 (zero-case refusal) gets a binary score: empty
prediction = 1.0, non-empty = 0.0 with the returned items logged as
hallucinations.

Usage:
    python eval.py --benchmark benchmark_v2_pilot_v2_300q.json \\
                   --predictions predictions.json \\
                   --output results.json [--tolerance-days 1]
到时候王浩会给我们写。然后你给我讲一下，这里一大点评完的逻辑是什么？
Predictions JSON format:
    {"predictions": [
        {"qid": "t01_q001", "answer": [{"firm_ticker":"MSFT", ...}, ...]},
        {"qid": "t01_q002", "answer": [...]},
        ...
    ]}

Or alternatively a flat dict:
    {"t01_q001": [{...}, {...}], "t01_q002": [...]}
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

try:
    from dateutil.parser import parse as dt_parse
except ImportError:
    dt_parse = None  # fall back to ISO-only parsing


# ============================== NORMALIZATION ==============================

def parse_date(s):
    """Parse a date string into a date object. Returns None if unparseable."""
    if s is None or s == '':
        return None
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    s = str(s).strip()
    # Try ISO first (no dependency)
    iso_match = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})', s)
    if iso_match:
        try:
            return date(int(iso_match.group(1)), int(iso_match.group(2)),
                       int(iso_match.group(3)))
        except ValueError:
            pass
    if dt_parse is not None:
        try:
            return dt_parse(s, fuzzy=False).date()
        except (ValueError, TypeError):
            return None
    return None


def normalize_str(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def normalize_tuple(t):
    """Normalize a single tuple (GT or prediction) for comparison."""
    if not isinstance(t, dict):
        return None
    firm = normalize_str(t.get('firm_ticker'))
    if firm:
        firm = firm.upper()
    et = normalize_str(t.get('event_type'))
    if et:
        et = et.lower().replace(' ', '_').replace('&', '&')
    role = normalize_str(t.get('role'))
    if role:
        role = role.lower()
    cp = normalize_str(t.get('counterparty_ticker'))
    if cp:
        cp = cp.upper()
    return {
        'firm_ticker': firm,
        'event_type': et,
        'anchor_date': parse_date(t.get('anchor_date')),
        'role': role,
        'counterparty_ticker': cp,
    }


def parse_predictions_loose(raw):
    """Robust prediction parser. Accepts:
       - list of tuples directly
       - dict with 'answer' / 'predictions' / 'output' keys
       - JSON string with surrounding noise (extracts first [...] block)
       Returns list (possibly empty) of normalized tuples.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        # Strip markdown fences and try to find a JSON array
        s = raw.strip()
        # Remove ```json ... ``` fences
        s = re.sub(r'^```(?:json)?\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
        # Find outermost [...]
        first_bracket = s.find('[')
        last_bracket = s.rfind(']')
        if first_bracket >= 0 and last_bracket > first_bracket:
            s = s[first_bracket:last_bracket + 1]
        try:
            raw = json.loads(s)
        except Exception:
            return []
    if isinstance(raw, dict):
        for key in ('answer', 'predictions', 'output', 'result'):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for t in raw:
        nt = normalize_tuple(t)
        if nt is not None:
            out.append(nt)
    return out


# ============================== MATCHING ==============================

def matches(gt: dict, pred: dict, tolerance_days: int) -> bool:
    """Tuple match: firm + event_type required; date within tolerance;
    role and counterparty required if GT has them populated."""
    if gt['firm_ticker'] != pred['firm_ticker']:
        return False
    if gt['event_type'] != pred['event_type']:
        return False
    g_date = gt['anchor_date']
    p_date = pred['anchor_date']
    if g_date is None and p_date is None:
        pass
    elif g_date is None or p_date is None:
        return False
    elif abs((g_date - p_date).days) > tolerance_days:
        return False
    if gt['role'] is not None and pred['role'] != gt['role']:
        return False
    if (gt['counterparty_ticker'] is not None
            and pred['counterparty_ticker'] != gt['counterparty_ticker']):
        return False
    return True


def score_question(question, pred_tuples, tolerance_days):
    """Score one question. Returns dict with recall, match counts, and
    diagnostic info. Special-cases T26 (empty GT)."""
    template = question['template_id']
    gt_tuples = [normalize_tuple(t) for t in question.get('answer', [])]
    gt_tuples = [t for t in gt_tuples if t]

    if template == 'T26':
        # Empty GT — refusal expected
        is_correct = len(pred_tuples) == 0
        return {
            'qid': question['qid'],
            'template_id': template,
            'gt_size': 0,
            'pred_size': len(pred_tuples),
            'matched': 0,
            'missed': 0,
            'extras': len(pred_tuples),
            'recall': 1.0 if is_correct else 0.0,
            'refusal_correct': is_correct,
            'hallucinated': pred_tuples if not is_correct else [],
        }

    # Standard recall: for each GT tuple, find a matching pred tuple (set semantics)
    # Each pred can only satisfy one GT (no double-counting).
    matched = 0
    consumed_pred_indices = set()
    matched_gt_indices = []
    for gi, gt in enumerate(gt_tuples):
        for pi, pred in enumerate(pred_tuples):
            if pi in consumed_pred_indices:
                continue
            if matches(gt, pred, tolerance_days):
                matched += 1
                consumed_pred_indices.add(pi)
                matched_gt_indices.append(gi)
                break

    n_gt = len(gt_tuples)
    n_pred = len(pred_tuples)
    extras = n_pred - matched
    missed = n_gt - matched
    recall = matched / n_gt if n_gt > 0 else None

    return {
        'qid': question['qid'],
        'template_id': template,
        'gt_size': n_gt,
        'pred_size': n_pred,
        'matched': matched,
        'missed': missed,
        'extras': extras,
        'recall': recall,
    }


# ============================== AGGREGATION ==============================

def aggregate(per_q_results):
    by_template = defaultdict(list)
    for r in per_q_results:
        by_template[r['template_id']].append(r)

    summary = {}
    for tid, rows in sorted(by_template.items()):
        if tid == 'T26':
            n = len(rows)
            n_correct = sum(1 for r in rows if r.get('refusal_correct'))
            avg_hallucinations = (
                sum(r['extras'] for r in rows if not r.get('refusal_correct'))
                / max(n - n_correct, 1)
            ) if (n - n_correct) > 0 else 0
            summary[tid] = {
                'n_questions': n,
                'refusal_accuracy': n_correct / n if n else 0,
                'n_correct_refusals': n_correct,
                'n_hallucinated': n - n_correct,
                'avg_hallucinations_when_failed': avg_hallucinations,
            }
        else:
            recalls = [r['recall'] for r in rows if r['recall'] is not None]
            macro = sum(recalls) / len(recalls) if recalls else 0
            total_matched = sum(r['matched'] for r in rows)
            total_gt = sum(r['gt_size'] for r in rows)
            micro = total_matched / total_gt if total_gt else 0
            total_extras = sum(r['extras'] for r in rows)
            summary[tid] = {
                'n_questions': len(rows),
                'macro_recall': macro,
                'micro_recall': micro,
                'total_matched': total_matched,
                'total_gt': total_gt,
                'total_extras_diagnostic': total_extras,
            }

    # Overall
    non_t26 = [r for r in per_q_results if r['template_id'] != 'T26']
    if non_t26:
        recalls = [r['recall'] for r in non_t26 if r['recall'] is not None]
        overall_macro = sum(recalls) / len(recalls) if recalls else 0
        total_matched = sum(r['matched'] for r in non_t26)
        total_gt = sum(r['gt_size'] for r in non_t26)
        overall_micro = total_matched / total_gt if total_gt else 0
    else:
        overall_macro = overall_micro = 0

    summary['_overall_non_T26'] = {
        'n_questions': len(non_t26),
        'macro_recall': overall_macro,
        'micro_recall': overall_micro,
    }
    return summary


# ============================== I/O ==============================

def load_predictions(path):
    """Returns dict: qid → raw prediction (list or string)."""
    raw = json.load(open(path))
    if isinstance(raw, dict):
        if 'predictions' in raw and isinstance(raw['predictions'], list):
            return {p['qid']: p.get('answer', p.get('output', [])) for p in raw['predictions']}
        return raw
    if isinstance(raw, list):
        return {p['qid']: p.get('answer', p.get('output', [])) for p in raw}
    raise ValueError(f'Unsupported predictions format in {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--output', default=None,
                    help='Write detailed results JSON here. If omitted, summary only.')
    ap.add_argument('--tolerance-days', type=int, default=1)
    args = ap.parse_args()

    bench = json.load(open(args.benchmark))
    questions = bench['questions']
    preds_by_qid = load_predictions(args.predictions)

    per_q = []
    n_missing = 0
    n_parse_failed = 0
    for q in questions:
        qid = q['qid']
        raw_pred = preds_by_qid.get(qid)
        if raw_pred is None:
            n_missing += 1
            pred_tuples = []
        else:
            pred_tuples = parse_predictions_loose(raw_pred)
            if not pred_tuples and (
                isinstance(raw_pred, list) and raw_pred
                or isinstance(raw_pred, str) and raw_pred.strip()
            ):
                # Non-empty input but parse failed (or all tuples invalid)
                if not isinstance(raw_pred, list) or not all(isinstance(x, dict) for x in raw_pred):
                    n_parse_failed += 1
        per_q.append(score_question(q, pred_tuples, args.tolerance_days))

    summary = aggregate(per_q)

    print(f'Benchmark: {bench.get("benchmark_version", "unknown")}')
    print(f'Tolerance: ±{args.tolerance_days} day(s)')
    print(f'Questions: {len(questions)}, predictions: {len(preds_by_qid)}, '
          f'missing: {n_missing}, parse-failed: {n_parse_failed}')
    print('\n=== Per-template ===')
    for tid, s in summary.items():
        if tid == '_overall_non_T26':
            continue
        if tid == 'T26':
            print(f'  {tid:5s} n={s["n_questions"]:3d}  '
                  f'refusal_acc={s["refusal_accuracy"]:.3f}  '
                  f'n_hallucinated={s["n_hallucinated"]}  '
                  f'avg_halluc_size={s["avg_hallucinations_when_failed"]:.1f}')
        else:
            print(f'  {tid:5s} n={s["n_questions"]:3d}  '
                  f'macro_recall={s["macro_recall"]:.3f}  '
                  f'micro_recall={s["micro_recall"]:.3f}  '
                  f'matched={s["total_matched"]}/{s["total_gt"]}  '
                  f'extras_diag={s["total_extras_diagnostic"]}')
    o = summary['_overall_non_T26']
    print(f'\nOverall (non-T26): macro_recall={o["macro_recall"]:.3f}  '
          f'micro_recall={o["micro_recall"]:.3f}  ({o["n_questions"]} questions)')

    if args.output:
        out = {
            'benchmark_version': bench.get('benchmark_version'),
            'evaluated_at': datetime.now().isoformat(),
            'tolerance_days': args.tolerance_days,
            'n_questions': len(questions),
            'n_predictions': len(preds_by_qid),
            'n_missing': n_missing,
            'n_parse_failed': n_parse_failed,
            'summary': summary,
            'per_question': per_q,
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'\nDetailed results written to {args.output}')


if __name__ == '__main__':
    main()
