#!/usr/bin/env python3
# QO-Bench (Zhang et al., 2026) — code/data released under MIT/CC-BY-NC-4.0.
# See LICENSE and DATA_LICENSE at bundle root for terms.
"""Build DuckDB events DB v2 — schema-honest union-find linking.

Two-stage canonicalization:
  Stage 1: Firm canonical (DSU on extraction's identity fields only)
           — atoms: ticker, cik, lei, normalized_name
           — no suffix stripping, no external dict, no fuzzy matching
  Stage 2: Event canonical (DSU on multiple identity keys)
           — atoms include: cluster_key_hint, (type,firm,date) tuple,
             M&A (acq,tgt,announce_date), (acq,tgt,deal_value_bucket),
             exec_change (firm,person,action), IPO (issuer,price,exchange),
             stock_split (firm,ratio,effective_date)
  Stage 3: M&A deal chain — links announce/complete/cancel via shared keys
  Stage 4: Exec change person chain — links via firm+person across records
  Stage 5: Field merge per cluster (carried over from v1)
  Stage 6: DuckDB insert (new tables: firm_canonical, deals)
  Stage 7: Audit log of edges by source

The pipeline uses ONLY schema-emitted fields from the LLM. No external
dictionaries (corpus ticker mapping disabled). No name stripping or
fuzzy matching. This isolates the paradigm's intrinsic linking capability.
"""

import argparse, json, re, sys
import duckdb
from collections import defaultdict, Counter
from datetime import date, timedelta
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════
# Disjoint Set Union
# ════════════════════════════════════════════════════════════════════════════

class DSU:
    def __init__(self):
        self.parent = {}
        self.size = defaultdict(int)

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        # iterative path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # union by size
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def parse_date(d):
    if d is None:
        return None
    if isinstance(d, str):
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", d)
        if not m:
            return None
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None
    return None


def date_value(dv):
    if not isinstance(dv, dict):
        return None, None, None
    return parse_date(dv.get("value")), dv.get("precision"), dv.get("text")


def money_tuple(m):
    if not isinstance(m, dict):
        return None, None, None
    return m.get("amount"), m.get("currency"), m.get("scale")


def deal_value_bucket(money_dict, bucket_pct=0.05):
    """Round to 5% buckets so $68B and $68.7B and $72B all distinguish but
    $69.0B and $69.3B collapse. None-safe."""
    if not isinstance(money_dict, dict):
        return None
    amt = money_dict.get("amount")
    if amt is None:
        return None
    scale = money_dict.get("scale") or "units"
    multiplier = {"units": 1, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}.get(scale, 1)
    val = amt * multiplier
    if val <= 0:
        return None
    # Round to nearest 5% bucket
    import math
    log_val = math.log10(val)
    bucket = round(val / (10**(int(log_val) - 1)) * bucket_pct) * (10**(int(log_val) - 1)) / bucket_pct
    return f"~{bucket:.2e}_{money_dict.get('currency') or 'USD'}"


def company_identity_atoms(c):
    """Extract identity atoms from a CompanyRef dict. Returns list of (type, value)."""
    out = []
    if not isinstance(c, dict):
        return out
    for fld in ("ticker", "cik", "lei", "normalized_name"):
        v = c.get(fld)
        if v and isinstance(v, str) and v.strip():
            out.append((fld, v.strip().lower() if fld == "normalized_name" else v.strip().upper() if fld == "ticker" else v.strip()))
    return out


# Precision rankings for date merging
PRECISION_ORDER = {"day": 4, "month": 3, "quarter": 2, "year": 1,
                   "approximate": 1, "unknown": 0, None: 0}


def merge_date(records, key_path):
    best = None
    best_prec = -1
    best_article_date = None
    for r, art_date in records:
        v = r
        for p in key_path.split("."):
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if not isinstance(v, dict):
            continue
        prec = PRECISION_ORDER.get(v.get("precision"), 0)
        d, p, t = date_value(v)
        if d is None:
            continue
        if prec > best_prec or (prec == best_prec and (best_article_date is None or (art_date and art_date < best_article_date))):
            best = (d, p, t)
            best_prec = prec
            best_article_date = art_date
    return best or (None, None, None)


def merge_money(records, key_path):
    candidates = []
    for r, art_date in records:
        v = r
        for p in key_path.split("."):
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if not isinstance(v, dict):
            continue
        a, c, s = money_tuple(v)
        if a is not None:
            candidates.append((art_date or date.min, a, c, s))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, a, c, s = candidates[0]
    return a, c, s


def merge_simple(records, key_path, mode="majority"):
    vals = []
    for r, art_date in records:
        v = r
        for p in key_path.split("."):
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None and v not in ("unknown", "", []):
            vals.append((v, art_date or date.min))
    if not vals:
        return None
    if mode == "any_true":
        for v, _ in vals:
            if v is True:
                return True
        return False if any(v is False for v, _ in vals) else None
    if mode == "latest":
        vals.sort(key=lambda x: x[1], reverse=True)
        return vals[0][0]
    if mode == "longest":
        vals.sort(key=lambda x: (len(str(x[0])), x[1]), reverse=True)
        return vals[0][0]
    counter = Counter(v for v, _ in vals)
    return counter.most_common(1)[0][0]


def merge_company_list(records, key_path):
    seen = {}
    for r, _ in records:
        v = r
        for p in key_path.split("."):
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if not isinstance(v, list):
            continue
        for c in v:
            if not isinstance(c, dict):
                continue
            nn = c.get("normalized_name")
            if not nn:
                continue
            if nn not in seen:
                seen[nn] = dict(c)
            else:
                for fld in ("ticker", "cik", "lei", "exchange", "country"):
                    if not seen[nn].get(fld) and c.get(fld):
                        seen[nn][fld] = c[fld]
    return list(seen.values())


# ════════════════════════════════════════════════════════════════════════════
# Stage 1: Firm canonical (DSU on identity atoms)
# ════════════════════════════════════════════════════════════════════════════

def stage1_firm_canonical(article_records, audit):
    """Build firm DSU. Each unique identity atom is a node; CompanyRef instances
    union their atoms. Returns dict: identity_atom -> firm_id."""
    print("Stage 1: Firm canonical (DSU on extraction-only identity fields)")
    dsu = DSU()
    n_companyrefs = 0
    n_atoms_per_ref = Counter()
    by_atom_type = defaultdict(int)

    for rec in article_records:
        e = rec["event"]
        d = e.get("details") or {}
        all_company_refs = []
        all_company_refs.extend(e.get("primary_companies") or [])
        for fld in ("acquirers", "targets", "sellers", "advisors"):
            for c in (d.get(fld) or []):
                if isinstance(c, dict):
                    all_company_refs.append(c)
        for sub in ("company", "issuer", "parent_company"):
            sub_v = d.get(sub)
            if isinstance(sub_v, dict):
                all_company_refs.append(sub_v)

        for c in all_company_refs:
            if not isinstance(c, dict):
                continue
            n_companyrefs += 1
            atoms = company_identity_atoms(c)
            n_atoms_per_ref[len(atoms)] += 1
            if not atoms:
                continue
            for t, _ in atoms:
                by_atom_type[t] += 1
            first = f"{atoms[0][0]}:{atoms[0][1]}"
            for t, v in atoms[1:]:
                dsu.union(first, f"{t}:{v}")

    # Build firm_id mapping
    atom_to_firm_id = {}
    firm_id_size = Counter()
    for k in list(dsu.parent.keys()):
        root = dsu.find(k)
        atom_to_firm_id[k] = root
        firm_id_size[root] += 1

    audit["firm_canonical"] = {
        "n_companyref_instances": n_companyrefs,
        "n_unique_atoms": len(atom_to_firm_id),
        "n_unique_firms": len(set(atom_to_firm_id.values())),
        "atoms_per_companyref_distribution": dict(n_atoms_per_ref),
        "atom_type_population": dict(by_atom_type),
        "top_firms_by_alias_count": [
            (firm_id, count) for firm_id, count in firm_id_size.most_common(20)
        ],
    }
    print(f"  CompanyRef instances:    {n_companyrefs:,}")
    print(f"  Unique identity atoms:   {len(atom_to_firm_id):,}")
    print(f"  Unique firms (DSU roots): {len(set(atom_to_firm_id.values())):,}")
    print(f"  Atom type population: {dict(by_atom_type)}")

    return atom_to_firm_id


def lookup_firm_id(c, atom_to_firm_id):
    """Resolve a CompanyRef dict to its firm_id, or None if no atom available."""
    if not isinstance(c, dict):
        return None
    atoms = company_identity_atoms(c)
    for t, v in atoms:
        key = f"{t}:{v}"
        if key in atom_to_firm_id:
            return atom_to_firm_id[key]
    return None


def primary_firm_id(event, atom_to_firm_id):
    """Resolve the primary firm for an event record."""
    d = event.get("details") or {}
    et = event.get("event_type", "")
    candidates = []
    if et.startswith("M&A"):
        for c in (d.get("acquirers") or []):
            candidates.append(c)
        for c in (d.get("targets") or []):
            candidates.append(c)
    elif et in ("CEO_change", "CFO_change"):
        c = d.get("company")
        if isinstance(c, dict):
            candidates.insert(0, c)
    elif et == "IPO":
        c = d.get("issuer")
        if isinstance(c, dict):
            candidates.insert(0, c)
    elif et == "Stock_split":
        c = d.get("company")
        if isinstance(c, dict):
            candidates.insert(0, c)
    candidates.extend(event.get("primary_companies") or [])
    for c in candidates:
        fid = lookup_firm_id(c, atom_to_firm_id)
        if fid:
            return fid
    return None


# ════════════════════════════════════════════════════════════════════════════
# Stage 2: Event canonical (DSU on multiple identity keys)
# ════════════════════════════════════════════════════════════════════════════

def event_identity_keys(record, atom_to_firm_id):
    """Compute the set of identity keys for a single article-level event record.
    Each key, if shared with another record's key, triggers union."""
    e = record["event"]
    d = e.get("details") or {}
    et = e.get("event_type", "")
    keys = set()
    edge_sources = []

    # K1: explicit cluster_key_hint (when LLM populated it)
    ck = e.get("duplicate_cluster_key_hint")
    if isinstance(ck, str) and ck.strip():
        keys.add(("cluster_key", ck.strip().lower()))
        edge_sources.append("cluster_key_hint")

    firm_id = primary_firm_id(e, atom_to_firm_id)
    if not firm_id:
        return keys, edge_sources

    # K2: (event_type, firm_id, event_date_value)
    ed_d, _, _ = date_value(e.get("event_date"))
    if ed_d:
        keys.add(("type_firm_date", et, firm_id, ed_d.isoformat()))
        edge_sources.append("type_firm_date")

    # K3+ event-type specific signatures
    if et.startswith("M&A"):
        acq_fids = sorted({lookup_firm_id(c, atom_to_firm_id) for c in (d.get("acquirers") or []) if isinstance(c, dict)} - {None})
        tgt_fids = sorted({lookup_firm_id(c, atom_to_firm_id) for c in (d.get("targets") or []) if isinstance(c, dict)} - {None})

        # K3: (event_type, acquirer, target, announcement_date)
        #     event_type included so M&A_announce / M&A_complete records of the
        #     SAME deal stay as separate event clusters (Stage 3 cross-links them).
        ann_d, _, _ = date_value(d.get("announcement_date"))
        if acq_fids and tgt_fids and ann_d:
            for a in acq_fids:
                for t in tgt_fids:
                    keys.add(("ma_acq_tgt_ann", et, a, t, ann_d.isoformat()))
                    edge_sources.append("ma_acq_tgt_announcement")

        # K4: (event_type, acquirer, target, deal_value_bucket)
        dvb = deal_value_bucket(d.get("deal_value"))
        if acq_fids and tgt_fids and dvb:
            for a in acq_fids:
                for t in tgt_fids:
                    keys.add(("ma_acq_tgt_value", et, a, t, dvb))
                    edge_sources.append("ma_acq_tgt_deal_value")

        # K5: (event_type, acquirer, target, closing/signing_date)
        close_d, _, _ = date_value(d.get("closing_date"))
        sign_d, _, _ = date_value(d.get("signing_date"))
        for anchor_d, label in [(close_d, "closing"), (sign_d, "signing")]:
            if acq_fids and tgt_fids and anchor_d:
                for a in acq_fids:
                    for t in tgt_fids:
                        keys.add((f"ma_acq_tgt_{label}", et, a, t, anchor_d.isoformat()))
                        edge_sources.append(f"ma_acq_tgt_{label}_date")

    elif et in ("CEO_change", "CFO_change"):
        person = d.get("person")
        if isinstance(person, dict):
            pn = person.get("normalized_name") or (person.get("full_name") or "").lower().replace(" ", "_")
            action = d.get("action")
            if pn and action:
                keys.add(("exec_firm_person_action", et, firm_id, pn, action))
                edge_sources.append("exec_firm_person_action")
            if pn:
                keys.add(("exec_firm_person", et, firm_id, pn))
                edge_sources.append("exec_firm_person")
        # Effective / announcement date variants
        for date_field in ("announcement_date", "effective_date", "departure_date", "appointment_date"):
            dd, _, _ = date_value(d.get(date_field))
            if dd:
                keys.add(("exec_firm_anydate", et, firm_id, dd.isoformat()))
                edge_sources.append(f"exec_{date_field}")

    elif et == "IPO":
        offer_price = d.get("offer_price")
        if isinstance(offer_price, dict) and offer_price.get("amount"):
            keys.add(("ipo_firm_price", firm_id, offer_price.get("amount"), offer_price.get("currency") or "USD"))
            edge_sources.append("ipo_firm_price")
        for date_field in ("pricing_date", "first_trading_date", "registration_filing_date"):
            dd, _, _ = date_value(d.get(date_field))
            if dd:
                keys.add(("ipo_firm_anydate", firm_id, dd.isoformat()))
                edge_sources.append(f"ipo_{date_field}")

    elif et == "Stock_split":
        ratio = d.get("ratio") or {}
        ratio_text = ratio.get("ratio_text") if isinstance(ratio, dict) else None
        if ratio_text:
            keys.add(("split_firm_ratio", firm_id, str(ratio_text).strip()))
            edge_sources.append("split_firm_ratio")
        for date_field in ("effective_date", "announcement_date", "record_date"):
            dd, _, _ = date_value(d.get(date_field))
            if dd:
                keys.add(("split_firm_anydate", firm_id, dd.isoformat()))
                edge_sources.append(f"split_{date_field}")

    return keys, edge_sources


def stage2_event_canonical(article_records, atom_to_firm_id, audit):
    """Build event DSU over article-level records using multi-key identities."""
    print("Stage 2: Event canonical (multi-key DSU)")
    dsu = DSU()
    edge_source_counts = Counter()
    record_ids = []
    record_keys_list = []

    for i, rec in enumerate(article_records):
        rid = f"r{i}"
        record_ids.append(rid)
        keys, sources = event_identity_keys(rec, atom_to_firm_id)
        record_keys_list.append(keys)
        for s in sources:
            edge_source_counts[s] += 1

    # Build per-key inverted index: key → list of record_ids
    key_to_records = defaultdict(list)
    for i, keys in enumerate(record_keys_list):
        for k in keys:
            key_to_records[k].append(record_ids[i])

    # Union records sharing any key
    n_edges = 0
    for k, members in key_to_records.items():
        if len(members) < 2:
            continue
        root = members[0]
        for m in members[1:]:
            if dsu.union(root, m):
                n_edges += 1

    # Make sure every record has a DSU node (singletons)
    for rid in record_ids:
        dsu.find(rid)

    # Resolve record → cluster_root
    record_to_cluster = {rid: dsu.find(rid) for rid in record_ids}
    clusters = defaultdict(list)
    for rid, root in record_to_cluster.items():
        clusters[root].append(int(rid[1:]))

    audit["event_canonical"] = {
        "n_records": len(record_ids),
        "n_clusters": len(clusters),
        "reduction_pct": (1 - len(clusters) / max(len(record_ids), 1)) * 100,
        "edge_source_population": dict(edge_source_counts),
        "n_union_edges": n_edges,
        "cluster_size_distribution": dict(Counter(len(m) for m in clusters.values())),
    }
    print(f"  Article-level records:   {len(record_ids):,}")
    print(f"  Identity keys produced:  {sum(len(k) for k in record_keys_list):,}")
    print(f"  Clusters (unique events): {len(clusters):,}")
    print(f"  Reduction: {len(record_ids)} → {len(clusters)} ({audit['event_canonical']['reduction_pct']:.1f}% deduped)")
    print(f"  Edges by source: {dict(edge_source_counts)}")

    return clusters, record_to_cluster


# ════════════════════════════════════════════════════════════════════════════
# Stage 3: M&A deal chain
# ════════════════════════════════════════════════════════════════════════════

def stage3_deal_chain(clusters, article_records, atom_to_firm_id, audit):
    """For each M&A cluster, identify (acq,tgt) signature; group clusters into deals."""
    print("Stage 3: M&A deal chain")
    deal_dsu = DSU()
    cluster_to_signature = {}

    for cluster_root, member_idxs in clusters.items():
        signatures = set()
        for idx in member_idxs:
            e = article_records[idx]["event"]
            et = e.get("event_type", "")
            if not et.startswith("M&A"):
                continue
            d = e.get("details") or {}
            acq_fids = tuple(sorted({lookup_firm_id(c, atom_to_firm_id) for c in (d.get("acquirers") or []) if isinstance(c, dict)} - {None}))
            tgt_fids = tuple(sorted({lookup_firm_id(c, atom_to_firm_id) for c in (d.get("targets") or []) if isinstance(c, dict)} - {None}))
            ann_d, _, _ = date_value(d.get("announcement_date"))
            ev_d, _, _ = date_value(e.get("event_date"))
            # cluster_key_hint as deal-level signature (cross event_type)
            ck = e.get("duplicate_cluster_key_hint")
            if isinstance(ck, str) and ck.strip():
                signatures.add(("ck", ck.strip().lower()))
            if acq_fids and tgt_fids:
                if ann_d:
                    signatures.add(("ann", acq_fids, tgt_fids, ann_d.isoformat()))
                if ev_d:
                    signatures.add(("ev", acq_fids, tgt_fids, ev_d.isoformat()))
                # Deal_value as cross-stage signature (announce + complete may both report)
                dvb = deal_value_bucket(d.get("deal_value"))
                if dvb:
                    signatures.add(("dv", acq_fids, tgt_fids, dvb))
        if signatures:
            cluster_to_signature[cluster_root] = signatures

    # Build sig → cluster index, then union clusters sharing any signature
    sig_to_clusters = defaultdict(list)
    for cl, sigs in cluster_to_signature.items():
        for s in sigs:
            sig_to_clusters[s].append(cl)
    n_deal_edges = 0
    for s, cls in sig_to_clusters.items():
        if len(cls) < 2:
            continue
        root = cls[0]
        for c in cls[1:]:
            if deal_dsu.union(root, c):
                n_deal_edges += 1
    for cl in clusters:
        deal_dsu.find(cl)

    # cluster → deal_id
    deal_map = {cl: deal_dsu.find(cl) for cl in clusters}
    deals_with_chains = defaultdict(set)
    for cl, did in deal_map.items():
        if cl in cluster_to_signature:
            deals_with_chains[did].add(cl)

    audit["deal_chain"] = {
        "n_ma_clusters": len(cluster_to_signature),
        "n_deals": len(deals_with_chains),
        "n_multi_cluster_deals": sum(1 for s in deals_with_chains.values() if len(s) > 1),
        "n_deal_edges_added": n_deal_edges,
    }
    print(f"  M&A clusters:            {len(cluster_to_signature):,}")
    print(f"  Deals (after chain merge): {len(deals_with_chains):,}")
    print(f"  Multi-stage deals (≥2 clusters): {audit['deal_chain']['n_multi_cluster_deals']:,}")

    return deal_map


# ════════════════════════════════════════════════════════════════════════════
# Stage 6: DDL + Insert
# ════════════════════════════════════════════════════════════════════════════

DDL = """
CREATE TABLE IF NOT EXISTS firm_canonical (
    firm_id           VARCHAR PRIMARY KEY,
    canonical_ticker  VARCHAR,
    canonical_name    VARCHAR,
    n_aliases         INTEGER
);

CREATE TABLE IF NOT EXISTS firm_alias (
    firm_id           VARCHAR,
    atom_type         VARCHAR,    -- ticker | cik | lei | normalized_name
    atom_value        VARCHAR
);

CREATE TABLE IF NOT EXISTS events (
    event_id              VARCHAR PRIMARY KEY,
    event_type            VARCHAR NOT NULL,
    firm_id               VARCHAR,                  -- canonical firm_id
    firm_ticker           VARCHAR,
    firm_name_canonical   VARCHAR,
    firm_name_raw         VARCHAR,
    event_date_value      DATE,
    event_date_precision  VARCHAR,
    event_date_text       VARCHAR,
    event_date_basis      VARCHAR,
    headline_summaries    JSON,
    geography             VARCHAR[],
    event_status          VARCHAR,
    raw_extraction_count  INTEGER,
    deal_id               VARCHAR
);

CREATE TABLE IF NOT EXISTS ma_details (
    event_id                          VARCHAR PRIMARY KEY,
    ma_event_subtype                  VARCHAR,
    deal_name                         VARCHAR,
    transaction_form                  VARCHAR,
    deal_status                       VARCHAR,
    consideration_type                VARCHAR,
    deal_value_amount                 DOUBLE,
    deal_value_currency               VARCHAR,
    deal_value_scale                  VARCHAR,
    per_share_price_amount            DOUBLE,
    announcement_date                 DATE,
    signing_date                      DATE,
    expected_closing_date             DATE,
    closing_date                      DATE,
    cluster_key_hint                  VARCHAR,
    strategic_rationale               VARCHAR
);

CREATE TABLE IF NOT EXISTS executive_change_details (
    event_id              VARCHAR PRIMARY KEY,
    role                  VARCHAR,
    action                VARCHAR,
    person_full_name      VARCHAR,
    person_normalized     VARCHAR,
    departure_reason      VARCHAR,
    appointment_source    VARCHAR,
    prior_title           VARCHAR,
    new_title             VARCHAR,
    announcement_date     DATE,
    effective_date        DATE,
    is_interim_or_acting  BOOLEAN
);

CREATE TABLE IF NOT EXISTS ipo_details (
    event_id              VARCHAR PRIMARY KEY,
    ipo_stage             VARCHAR,
    offering_type         VARCHAR,
    is_first_public_listing BOOLEAN,
    is_spac_ipo           BOOLEAN,
    exchange              VARCHAR,
    ticker                VARCHAR,
    pricing_date          DATE,
    first_trading_date    DATE,
    offer_price_amount    DOUBLE,
    offer_price_currency  VARCHAR
);

CREATE TABLE IF NOT EXISTS stock_split_details (
    event_id              VARCHAR PRIMARY KEY,
    split_type            VARCHAR,
    ratio_text            VARCHAR,
    new_shares            DOUBLE,
    old_shares            DOUBLE,
    share_count_multiplier DOUBLE,
    announcement_date     DATE,
    effective_date        DATE
);

CREATE TABLE IF NOT EXISTS event_companies (
    event_id          VARCHAR,
    role              VARCHAR,
    firm_id           VARCHAR,
    name              VARCHAR,
    normalized_name   VARCHAR,
    ticker            VARCHAR
);

CREATE TABLE IF NOT EXISTS event_sources (
    event_id          VARCHAR,
    article_id        VARCHAR,
    article_date      DATE,
    article_title     VARCHAR,
    article_url       VARCHAR
);

CREATE TABLE IF NOT EXISTS deals (
    deal_id               VARCHAR PRIMARY KEY,
    announce_event_id     VARCHAR,
    complete_event_id     VARCHAR,
    cancel_event_id       VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date_value);
CREATE INDEX IF NOT EXISTS idx_events_firm ON events(firm_ticker);
CREATE INDEX IF NOT EXISTS idx_events_firm_id ON events(firm_id);
CREATE INDEX IF NOT EXISTS idx_companies_role ON event_companies(role, event_id);
CREATE INDEX IF NOT EXISTS idx_companies_firm_id ON event_companies(firm_id);
CREATE INDEX IF NOT EXISTS idx_alias_value ON firm_alias(atom_type, atom_value);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-db", default="events_v2.duckdb")
    ap.add_argument("--audit-log", default="events_v2_audit.json")
    args = ap.parse_args()

    audit = {}
    in_path = Path(args.input)
    db_path = Path(args.output_db)
    if db_path.exists():
        db_path.unlink()

    # ── Load ──
    print(f"Loading from {in_path}...")
    article_records = []
    n_articles_ok = 0
    for line in in_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("ok") is not True:
            continue
        n_articles_ok += 1
        for e in r.get("parsed_events_all", []):
            if isinstance(e, dict):
                article_records.append({
                    "article_id": r["article_id"],
                    "article_date": parse_date(r.get("article_date")),
                    "article_title": r.get("title"),
                    "article_url": (e.get("source_article") or {}).get("url") if isinstance(e.get("source_article"), dict) else None,
                    "event": e,
                })
    print(f"  ok articles: {n_articles_ok}  event records: {len(article_records)}")
    audit["input"] = {"n_articles_ok": n_articles_ok, "n_event_records": len(article_records)}

    # ── Stage 1 ──
    atom_to_firm_id = stage1_firm_canonical(article_records, audit)

    # ── Stage 2 ──
    clusters, record_to_cluster = stage2_event_canonical(article_records, atom_to_firm_id, audit)

    # ── Stage 3 ──
    deal_map = stage3_deal_chain(clusters, article_records, atom_to_firm_id, audit)

    # ── DuckDB ──
    print(f"\nCreating DuckDB at {db_path}...")
    con = duckdb.connect(str(db_path))
    con.execute(DDL)

    # Build firm_canonical table
    firm_atoms_by_id = defaultdict(list)
    for atom, fid in atom_to_firm_id.items():
        atom_type, atom_value = atom.split(":", 1)
        firm_atoms_by_id[fid].append((atom_type, atom_value))

    firm_canonical_rows = []
    firm_alias_rows = []
    for fid, atoms in firm_atoms_by_id.items():
        # Pick canonical ticker (uppercase, longest) and canonical name
        tickers = [v for t, v in atoms if t == "ticker"]
        normed = [v for t, v in atoms if t == "normalized_name"]
        canonical_ticker = sorted(tickers, key=len, reverse=True)[0] if tickers else None
        canonical_name = sorted(normed, key=len, reverse=True)[0] if normed else None
        firm_canonical_rows.append((fid, canonical_ticker, canonical_name, len(atoms)))
        for t, v in atoms:
            firm_alias_rows.append((fid, t, v))
    if firm_canonical_rows:
        con.executemany("INSERT INTO firm_canonical VALUES (?,?,?,?)", firm_canonical_rows)
    if firm_alias_rows:
        con.executemany("INSERT INTO firm_alias VALUES (?,?,?)", firm_alias_rows)
    print(f"  firm_canonical: {len(firm_canonical_rows)}  firm_alias: {len(firm_alias_rows)}")

    # ── Stage 5: Merge clusters into event rows ──
    print(f"\nStage 5: Merging {len(clusters)} clusters and writing events...")
    event_rows = []
    ma_rows = []
    exec_rows = []
    ipo_rows = []
    split_rows = []
    company_rows = []
    source_rows = []
    deal_rows_dict = defaultdict(lambda: {"announce": None, "complete": None, "cancel": None})

    for cluster_root, member_idxs in clusters.items():
        members = [article_records[i] for i in member_idxs]
        member_records = [(m["event"], m["article_date"]) for m in members]
        first_ev = members[0]["event"]
        et = first_ev.get("event_type")

        event_id = f"evt_{abs(hash(cluster_root)) & 0xFFFFFFFFFFFF:012x}"
        deal_id = None
        if et and et.startswith("M&A"):
            deal_root = deal_map.get(cluster_root)
            if deal_root is not None:
                deal_id = f"deal_{abs(hash(deal_root)) & 0xFFFFFFFFFFFF:012x}"

        firm_id = primary_firm_id(first_ev, atom_to_firm_id)
        canonical_ticker = None
        canonical_name = None
        firm_raw_name = None
        if firm_id:
            atoms = firm_atoms_by_id.get(firm_id, [])
            tickers = [v for t, v in atoms if t == "ticker"]
            normed = [v for t, v in atoms if t == "normalized_name"]
            canonical_ticker = sorted(tickers, key=len, reverse=True)[0] if tickers else None
            canonical_name = sorted(normed, key=len, reverse=True)[0] if normed else None

        # Find raw firm name from member CompanyRefs
        for m, _ in member_records:
            d = m.get("details") or {}
            search_lists = [m.get("primary_companies") or [],
                           d.get("acquirers") or [], d.get("targets") or [],
                           [d.get("company")] if isinstance(d.get("company"), dict) else [],
                           [d.get("issuer")] if isinstance(d.get("issuer"), dict) else []]
            for lst in search_lists:
                for c in lst:
                    if isinstance(c, dict) and lookup_firm_id(c, atom_to_firm_id) == firm_id:
                        if c.get("name"):
                            firm_raw_name = c["name"]
                            break
                if firm_raw_name:
                    break
            if firm_raw_name:
                break

        ed, ep, etext = merge_date(member_records, "event_date")
        event_date_basis = merge_simple(member_records, "event_date_basis", "majority")
        event_status = merge_simple(member_records, "event_status", "latest")
        headlines = [h for m, _ in member_records for h in [m.get("headline_summary")] if h]
        geos = sorted({g for m, _ in member_records for g in (m.get("geography") or []) if g})

        event_rows.append((
            event_id, et, firm_id, canonical_ticker, canonical_name, firm_raw_name,
            ed, ep, etext, event_date_basis,
            json.dumps(headlines, ensure_ascii=False),
            geos, event_status, len(members), deal_id,
        ))

        # Per-event-type details
        if et and et.startswith("M&A"):
            ma_rows.append((
                event_id,
                merge_simple(member_records, "details.ma_event_subtype", "majority") or et,
                merge_simple(member_records, "details.deal_name", "longest"),
                merge_simple(member_records, "details.transaction_form", "majority"),
                merge_simple(member_records, "details.deal_status", "latest"),
                merge_simple(member_records, "details.consideration_type", "majority"),
                *merge_money(member_records, "details.deal_value"),
                merge_money(member_records, "details.per_share_price")[0],
                merge_date(member_records, "details.announcement_date")[0],
                merge_date(member_records, "details.signing_date")[0],
                merge_date(member_records, "details.expected_closing_date")[0],
                merge_date(member_records, "details.closing_date")[0],
                merge_simple(member_records, "duplicate_cluster_key_hint", "majority"),
                merge_simple(member_records, "details.strategic_rationale", "longest"),
            ))
            # Roles
            for role_field in ("acquirers", "targets", "sellers"):
                for c in merge_company_list(member_records, f"details.{role_field}"):
                    company_rows.append((
                        event_id, role_field.rstrip("s"),
                        lookup_firm_id(c, atom_to_firm_id),
                        c.get("name"),
                        c.get("normalized_name"),
                        c.get("ticker"),
                    ))
            # Track deal slot
            if deal_id:
                subtype = et.replace("M&A_", "")
                if subtype in ("announce", "complete", "cancel"):
                    if deal_rows_dict[deal_id][subtype] is None:
                        deal_rows_dict[deal_id][subtype] = event_id

        elif et in ("CEO_change", "CFO_change"):
            person = None
            for m, _ in member_records:
                p = (m.get("details") or {}).get("person")
                if isinstance(p, dict) and p.get("full_name"):
                    person = p; break
            exec_rows.append((
                event_id,
                "CEO" if et == "CEO_change" else "CFO",
                merge_simple(member_records, "details.action", "majority"),
                person.get("full_name") if person else None,
                person.get("normalized_name") if person else None,
                merge_simple(member_records, "details.departure_reason", "majority"),
                merge_simple(member_records, "details.appointment_source", "majority"),
                merge_simple(member_records, "details.prior_title", "longest"),
                merge_simple(member_records, "details.new_title", "longest"),
                merge_date(member_records, "details.announcement_date")[0],
                merge_date(member_records, "details.effective_date")[0],
                merge_simple(member_records, "details.is_interim_or_acting", "any_true"),
            ))

        elif et == "IPO":
            ipo_rows.append((
                event_id,
                merge_simple(member_records, "details.ipo_stage", "latest"),
                merge_simple(member_records, "details.offering_type", "majority"),
                merge_simple(member_records, "details.is_first_public_listing", "any_true"),
                merge_simple(member_records, "details.is_spac_ipo", "any_true"),
                merge_simple(member_records, "details.exchange", "majority"),
                merge_simple(member_records, "details.ticker", "majority"),
                merge_date(member_records, "details.pricing_date")[0],
                merge_date(member_records, "details.first_trading_date")[0],
                *merge_money(member_records, "details.offer_price")[:2],
            ))

        elif et == "Stock_split":
            split_rows.append((
                event_id,
                merge_simple(member_records, "details.split_type", "majority"),
                merge_simple(member_records, "details.ratio.ratio_text", "majority"),
                merge_simple(member_records, "details.ratio.new_shares", "majority"),
                merge_simple(member_records, "details.ratio.old_shares", "majority"),
                merge_simple(member_records, "details.ratio.share_count_multiplier", "majority"),
                merge_date(member_records, "details.announcement_date")[0],
                merge_date(member_records, "details.effective_date")[0],
            ))

        # Add primary_companies as junction rows (role=primary)
        for c in merge_company_list(member_records, "primary_companies"):
            company_rows.append((
                event_id, "primary",
                lookup_firm_id(c, atom_to_firm_id),
                c.get("name"),
                c.get("normalized_name"),
                c.get("ticker"),
            ))
        for m in members:
            source_rows.append((event_id, m["article_id"], m["article_date"], m["article_title"], m["article_url"]))

    # Deal rows
    deal_rows = [(did, slots["announce"], slots["complete"], slots["cancel"])
                 for did, slots in deal_rows_dict.items()]

    # Bulk insert
    print(f"  events:{len(event_rows)}  ma:{len(ma_rows)}  exec:{len(exec_rows)}  ipo:{len(ipo_rows)}  split:{len(split_rows)}")
    print(f"  companies:{len(company_rows)}  sources:{len(source_rows)}  deals:{len(deal_rows)}")
    con.executemany("INSERT INTO events VALUES (" + ",".join("?"*15) + ")", event_rows)
    if ma_rows: con.executemany("INSERT INTO ma_details VALUES (" + ",".join("?"*len(ma_rows[0])) + ")", ma_rows)
    if exec_rows: con.executemany("INSERT INTO executive_change_details VALUES (" + ",".join("?"*len(exec_rows[0])) + ")", exec_rows)
    if ipo_rows: con.executemany("INSERT INTO ipo_details VALUES (" + ",".join("?"*len(ipo_rows[0])) + ")", ipo_rows)
    if split_rows: con.executemany("INSERT INTO stock_split_details VALUES (" + ",".join("?"*len(split_rows[0])) + ")", split_rows)
    if company_rows: con.executemany("INSERT INTO event_companies VALUES (?,?,?,?,?,?)", company_rows)
    if source_rows: con.executemany("INSERT INTO event_sources VALUES (?,?,?,?,?)", source_rows)
    if deal_rows: con.executemany("INSERT INTO deals VALUES (?,?,?,?)", deal_rows)

    # Summary stats
    print("\n=== Build summary ===")
    counts = con.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type ORDER BY 2 DESC").fetchall()
    for et, n in counts:
        print(f"  {et}: {n:,}")
    total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    avg = con.execute("SELECT AVG(raw_extraction_count) FROM events").fetchone()[0]
    print(f"  TOTAL events: {total:,}  (was {len(article_records):,} records; {(1-total/len(article_records))*100:.1f}% deduped)")
    print(f"  avg extractions per dedupd event: {avg:.2f}")
    n_deals = con.execute("SELECT COUNT(*) FROM deals").fetchone()[0]
    n_multi_stage = con.execute("SELECT COUNT(*) FROM deals WHERE announce_event_id IS NOT NULL AND complete_event_id IS NOT NULL").fetchone()[0]
    print(f"  deals: {n_deals:,}  multi-stage (announce+complete): {n_multi_stage:,}")

    # Audit log
    Path(args.audit_log).write_text(json.dumps(audit, indent=2, default=str, ensure_ascii=False))
    print(f"\nAudit log: {args.audit_log}")
    print(f"DB written: {db_path}  ({db_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
