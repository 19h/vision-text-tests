#!/usr/bin/env python3
"""Repeated-measures analysis for E-BIND-1.

Calls are not independent items. This report groups passes by semantic item and performs
a deterministic hierarchical bootstrap over sampling page, then item. Legacy result files
without sampling-page metadata fall back to an item-cluster bootstrap and say so explicitly.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random

import provenance as V

ROOT = os.path.dirname(os.path.abspath(__file__))
PRESENT = ("C", "D", "W", "A", "P")
ABSENT = ("N", "W0", "D0", "P0")


def item_id(row):
    return row.get("item_id") or hashlib.sha256(
        f"{row.get('kind')}|{row.get('query')}".encode()).hexdigest()[:16]


def sampling_page(row):
    if row.get("sampling_page"):
        return row["sampling_page"]
    gold = row.get("gold")
    if isinstance(gold, (list, tuple)) and gold:
        return gold[0]
    return None


def quantile(values, p):
    if not values: return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi: return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def descriptive_metrics(rows):
    seconds = [float(r['seconds']) for r in rows
               if isinstance(r.get('seconds'), (int, float))]
    input_tokens, output_tokens, cached_tokens = [], [], []
    for row in rows:
        usage = ((row.get('response') or {}).get('usage') or {})
        if isinstance(usage.get('input_tokens'), (int, float)):
            input_tokens.append(float(usage['input_tokens']))
        if isinstance(usage.get('output_tokens'), (int, float)):
            output_tokens.append(float(usage['output_tokens']))
        if isinstance(usage.get('cached_input_tokens'), (int, float)):
            cached_tokens.append(float(usage['cached_input_tokens']))
    present = [r for r in rows if r.get('kind') == 'present']
    absent = [r for r in rows if r.get('kind') == 'absent']
    correct = sum(r.get('outcome') == 'C' for r in present)
    correct_absent = sum(r.get('outcome') == 'N' for r in absent)
    correct_block = [r for r in present if r.get('correct_block') is True]
    protocol_parsed = [r for r in rows if r.get('outcome') not in ('P', 'P0')]
    canonical_code = [r for r in present
                      if r.get('validator') and r.get('validator') != 'range']
    tag_checked = [r for r in rows if r.get('validator') in ('ok', 'tag_fail')]
    tag_pass = [r for r in tag_checked if r.get('validator') == 'ok']
    total_input = sum(input_tokens)
    return {
        'latency_seconds': {
            'n': len(seconds), 'p50': quantile(seconds, .5) if seconds else None,
            'p95': quantile(seconds, .95) if seconds else None,
        },
        'usage_tokens': {
            'calls_with_usage': len(input_tokens), 'input_total': total_input,
            'input_p50': quantile(input_tokens, .5) if input_tokens else None,
            'output_total': sum(output_tokens), 'cached_input_total': sum(cached_tokens),
        },
        'protocol_response_rate': len(protocol_parsed) / len(rows) if rows else None,
        'canonical_code_parse_rate': len(canonical_code) / len(present) if present else None,
        'tag_pass_given_tag_checked': len(tag_pass) / len(tag_checked) if tag_checked else None,
        'correct_block_rate_present': len(correct_block) / len(present) if present else None,
        'correct_record_given_correct_block': correct / len(correct_block) if correct_block else None,
        'geometric_attempts_if_iid': {
            'present_correct': len(present) / correct if correct else None,
            'absent_correct': len(absent) / correct_absent if correct_absent else None,
        },
        'successful_present_queries_per_recorded_input_token':
            correct / total_input if total_input else None,
    }


def wilson_interval(successes, n, z=1.959963984540054):
    if not n: return [float("nan"), float("nan")]
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def cluster_interval(rows, event, reps=5000, seed=0):
    """CI for mean per-item event probability, clustering calls within items/pages."""
    items = collections.defaultdict(list)
    pages = {}
    for row in rows:
        iid = item_id(row)
        items[iid].append(1.0 if row.get("outcome") == event else 0.0)
        pages[iid] = sampling_page(row)
    means = {iid: sum(v) / len(v) for iid, v in items.items()}
    point = sum(means.values()) / len(means) if means else float("nan")
    if not means:
        return {"rate": float("nan"), "ci95": [float("nan"), float("nan")],
                "n_items": 0, "n_calls": 0, "cluster": "none"}
    known_pages = all(pages[iid] is not None for iid in items)
    by_page = collections.defaultdict(list)
    for iid in items:
        by_page[pages[iid] if known_pages else iid].append(iid)
    page_keys = sorted(by_page, key=str)
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        vals = []
        for _ in page_keys:
            page = rng.choice(page_keys)
            candidates = by_page[page]
            for _ in candidates:
                vals.append(means[rng.choice(candidates)])
        samples.append(sum(vals) / len(vals))
    interval = [quantile(samples, 0.025), quantile(samples, 0.975)]
    cluster = "page_then_item" if known_pages else "item_only_legacy"
    # An empirical bootstrap cannot invent an unseen event and otherwise returns
    # [0,0] or [1,1]. Use the number of independent top-level clusters for a
    # boundary-safe Wilson interval; this is intentionally conservative for pages.
    if all(value == 0 for value in means.values()) or all(value == 1 for value in means.values()):
        boundary_successes = len(page_keys) if point == 1 else 0
        interval = wilson_interval(boundary_successes, len(page_keys))
        cluster += "+wilson_boundary"
    return {
        "rate": point,
        "ci95": interval,
        "n_items": len(items),
        "n_calls": len(rows),
        "cluster": cluster,
    }


def summarize(rows, bootstrap_reps=5000):
    groups = collections.defaultdict(list)
    include_arm = any(row.get('arm') for row in rows)
    for row in rows:
        key = ((row.get('arm', '?'), row.get("carrier", "image"), row.get("enc", "hex"))
               if include_arm else (row.get("carrier", "image"), row.get("enc", "hex")))
        groups[key].append(row)
    out = {}
    for key, rs in sorted(groups.items()):
        arm, carrier, enc = key if include_arm else (None, *key)
        record = {
            "calls": len(rs),
            "items": len({item_id(r) for r in rs}),
            "passes": sorted({r.get("rep") for r in rs if r.get("rep") is not None}),
            "call_counts": dict(collections.Counter(r.get("outcome") for r in rs)),
            "present": {}, "absent": {}, "descriptive": descriptive_metrics(rs),
        }
        for event in PRESENT:
            sub = [r for r in rs if r.get("kind") == "present"]
            record["present"][event] = cluster_interval(
                sub, event, bootstrap_reps, seed=int(hashlib.sha256(
                    f"{carrier}|{enc}|{event}".encode()).hexdigest()[:8], 16))
        for event in ABSENT:
            sub = [r for r in rs if r.get("kind") == "absent"]
            record["absent"][event] = cluster_interval(
                sub, event, bootstrap_reps, seed=int(hashlib.sha256(
                    f"{carrier}|{enc}|{event}".encode()).hexdigest()[:8], 16))
        out[f"{arm}/{carrier}/{enc}" if include_arm else f"{carrier}/{enc}"] = record
    return out


def paired_condition_difference(rows, kind, success, key, a, b, reps=5000,
                                predicate=lambda row: True):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    pages = {}
    for row in rows:
        if row.get("kind") != kind or not predicate(row): continue
        iid = item_id(row)
        by[iid][key(row)].append(1.0 if row.get("outcome") == success else 0.0)
        pages[iid] = sampling_page(row)
    diffs = {}
    for iid, carriers in by.items():
        if carriers.get(a) and carriers.get(b):
            diffs[iid] = (sum(carriers[a]) / len(carriers[a]) -
                          sum(carriers[b]) / len(carriers[b]))
    if not diffs:
        return {"difference": None, "ci95": [None, None], "n_paired_items": 0}
    known_pages = all(pages[iid] is not None for iid in diffs)
    by_page = collections.defaultdict(list)
    for iid in diffs:
        by_page[pages[iid] if known_pages else iid].append(iid)
    page_keys = sorted(by_page, key=str)
    rng = random.Random(0xEB1D)
    boot = []
    for _ in range(reps):
        values = []
        for _ in page_keys:
            page = rng.choice(page_keys)
            candidates = by_page[page]
            for _ in candidates:
                values.append(diffs[rng.choice(candidates)])
        boot.append(sum(values) / len(values))
    return {"difference": sum(diffs.values()) / len(diffs),
            "ci95": [quantile(boot, .025), quantile(boot, .975)],
            "n_paired_items": len(diffs), "n_page_clusters": len(page_keys),
            "cluster": "page_then_item" if known_pages else "item_only_legacy",
            "contrast": f"{a} - {b}"}


def paired_carrier_difference(rows, kind, success, a="image", b="text", reps=5000, arm=None):
    return paired_condition_difference(
        rows, kind, success, lambda row: row.get('carrier'), a, b, reps,
        predicate=(lambda row: arm is None or row.get('arm') == arm))


def markdown(report):
    lines = ["# E-BIND-1 repeated-measures analysis", "",
             "Rates are means of per-item call probabilities; intervals resample pages then items. "
             "All-zero/all-one endpoints use a boundary-safe Wilson interval over top-level clusters.", "",
             "| carrier/encoding | items | calls | C | W | N | W0 | clustering |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, g in report["groups"].items():
        def fmt(part, event):
            x = g[part][event]
            return f"{x['rate']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]" if x["n_items"] else "-"
        clusters = []
        for part in ("present", "absent"):
            cluster = next((x["cluster"] for x in g[part].values() if x["n_items"]), None)
            if cluster and cluster not in clusters: clusters.append(cluster)
        lines.append(f"| {name} | {g['items']} | {g['calls']} | {fmt('present','C')} | "
                     f"{fmt('present','W')} | {fmt('absent','N')} | {fmt('absent','W0')} | {' / '.join(clusters)} |")
    lines += ["", "Operational/descriptive metrics (retry figures assume IID attempts):", "",
              "| carrier/encoding | p50/p95 seconds | correct block | C given correct block | attempts/C | input tokens | C/input token |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, group in report['groups'].items():
        d = group['descriptive']; lat = d['latency_seconds']; usage = d['usage_tokens']
        def number(value, places=3):
            return '-' if value is None else f"{value:.{places}f}"
        latency = '-' if lat['p50'] is None else f"{lat['p50']:.1f}/{lat['p95']:.1f}"
        lines.append(f"| {name} | {latency} | {number(d['correct_block_rate_present'])} | "
                     f"{number(d['correct_record_given_correct_block'])} | "
                     f"{number(d['geometric_attempts_if_iid']['present_correct'], 2)} | "
                     f"{usage['input_total']:.0f} | "
                     f"{number(d['successful_present_queries_per_recorded_input_token'], 6)} |")
    for heading, section in (("Paired carrier contrasts", "paired_carrier_differences"),
                             ("Paired architecture contrasts", "paired_architecture_differences")):
        lines += ["", heading + ":", ""]
        for name, x in report.get(section, {}).items():
            if x["difference"] is None:
                lines.append(f"- {name}: unavailable (no paired items)")
            else:
                lines.append(f"- {name}: {x['difference']:+.3f} "
                             f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}], "
                             f"n={x['n_paired_items']} items, {x['cluster']}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--json-out")
    ap.add_argument("--markdown-out")
    a = ap.parse_args()
    rows = []
    sources = []
    archive_geometry = {}
    for name in a.files:
        path = os.path.abspath(name)
        with open(path) as f:
            blob = json.load(f)
        rows += V.result_rows(blob)
        source_manifest = blob.get('manifest') or {}
        for enc, geo in (source_manifest.get('archive_geometry') or {}).items():
            archive_geometry[(source_manifest.get('arm'), enc)] = geo
        sources.append({"path": os.path.relpath(path, ROOT), "sha256": V.sha256_file(path)})
    groups = summarize(rows, a.bootstrap_reps)
    for name, group in groups.items():
        parts = name.split('/')
        carrier, enc = parts[-2], parts[-1]
        arm = parts[0] if len(parts) == 3 else None
        geo = archive_geometry.get((arm, enc)) if carrier == 'image' else None
        if geo:
            c_rate = group['present']['C']['rate']
            group['goodput'] = {
                'archive_records_per_image_token': geo['records_per_image_token'],
                'successful_records_per_image_token': geo['records_per_image_token'] * c_rate,
                'archive_image_tokens_per_query': geo['image_tokens'],
            }
    arms = sorted({r.get('arm') for r in rows if r.get('arm')})
    contrasts = {}
    for arm in arms or [None]:
        prefix = f'{arm}_' if arm else ''
        contrasts[prefix + 'present_correct'] = paired_carrier_difference(
            rows, 'present', 'C', reps=a.bootstrap_reps, arm=arm)
        contrasts[prefix + 'absent_correct'] = paired_carrier_difference(
            rows, 'absent', 'N', reps=a.bootstrap_reps, arm=arm)
    architecture = {
        'arm_A_minus_B_present_correct_image_hex': paired_condition_difference(
            rows, 'present', 'C', lambda row: row.get('arm'), 'A', 'B', a.bootstrap_reps,
            predicate=lambda row: row.get('carrier') == 'image' and row.get('enc') == 'hex'),
        'arm_A_minus_B_absent_correct_image_hex': paired_condition_difference(
            rows, 'absent', 'N', lambda row: row.get('arm'), 'A', 'B', a.bootstrap_reps,
            predicate=lambda row: row.get('carrier') == 'image' and row.get('enc') == 'hex'),
        'arm_B_hex_minus_b32_present_correct': paired_condition_difference(
            rows, 'present', 'C', lambda row: row.get('enc'), 'hex', 'b32', a.bootstrap_reps,
            predicate=lambda row: row.get('arm') == 'B' and row.get('carrier') == 'image'),
    }
    report = {
        "schema_version": "ebind-analysis-v1",
        "sources": sources,
        "groups": groups,
        "paired_carrier_differences": contrasts,
        "paired_architecture_differences": architecture,
    }
    text = markdown(report)
    if a.json_out: V.dump_json(a.json_out, report)
    if a.markdown_out:
        with open(a.markdown_out, "w") as f: f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
