# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-NC-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Build Milvus filter expressions for date windows.

Supports two date formats discovered at runtime:
  - "yyyymmdd_int":  date is INT64 like 20150130
  - "iso_string":    date is VARCHAR like "2015-01-30"
"""
from __future__ import annotations

from datetime import date, datetime, timedelta


def _to_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def build_date_expr(field: str, w_start, w_end, pad_days: int = 30,
                    date_format: str = "yyyymmdd_int") -> str:
    """Return a Milvus boolean expr restricting `field` to [w_start - pad, w_end + pad].

    Args:
        field: Milvus field name (e.g., "date" or "anchor_date").
        w_start, w_end: ISO-format strings or date/datetime objects.
        pad_days: Symmetric padding around the window.
        date_format: "yyyymmdd_int" (INT64 YYYYMMDD) or "iso_string" (VARCHAR ISO).
    """
    start = _to_date(w_start) - timedelta(days=pad_days)
    end = _to_date(w_end) + timedelta(days=pad_days)
    if date_format == "yyyymmdd_int":
        s = int(start.strftime("%Y%m%d"))
        e = int(end.strftime("%Y%m%d"))
        return f"{field} >= {s} and {field} <= {e}"
    if date_format == "iso_string":
        return f'{field} >= "{start.isoformat()}" and {field} <= "{end.isoformat()}"'
    raise ValueError(f"Unknown date_format: {date_format!r}")
