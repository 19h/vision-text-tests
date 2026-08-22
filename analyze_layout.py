#!/usr/bin/env python3
"""Paired effects and equivalence intervals for layout_probe.py results."""
from __future__ import annotations

import argparse
import collections
import json
import random

import provenance as V
from analyze_encoding import paired_difference_interval


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round((len(xs) - 1) * p)))]


def paired_effect(rows, bit_count, factor, level_a, level_b, fixed, reps=10000):
    by_item = collections.defaultdict(dict)
    for r in rows:
        if r['bits'] != bit_count or any(r[k] != v for k, v in fixed.items()): continue
        by_item[r['item_id']][r[factor]] = 1.0 if r['value'] else 0.0
    diffs = [v[level_b] - v[level_a] for v in by_item.values()
             if level_a in v and level_b in v]
    if not diffs: return {'n': 0, 'difference': None, 'ci95': [None, None]}
    rng = random.Random(0x2B2)
    boot = [sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(reps)]
    return {'n': len(diffs), 'difference': sum(diffs) / len(diffs),
            'ci95': paired_difference_interval(diffs),
            'interval_method': 'bonferroni_clopper_pearson_discordant_cells',
            'bootstrap_ci95_diagnostic': [percentile(boot, .025), percentile(boot, .975)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('--equivalence-margin', type=float, default=.10)
    ap.add_argument('--bootstrap-reps', type=int, default=10000)
    ap.add_argument('--out')
    a = ap.parse_args()
    rows = V.load_result_rows(a.result)
    report = {'schema_version': 'layout-analysis-v1', 'margin': a.equivalence_margin,
              'source_sha256': V.sha256_file(a.result), 'effects': {}}
    for bits in (64, 128):
        for grouped in (False, True):
            name = f'{bits}b_delim_effect_grouped={grouped}'
            report['effects'][name] = paired_effect(
                rows, bits, 'delimited', False, True, {'grouped': grouped}, a.bootstrap_reps)
        for delimited in (False, True):
            name = f'{bits}b_group_effect_delimited={delimited}'
            report['effects'][name] = paired_effect(
                rows, bits, 'grouped', False, True, {'delimited': delimited}, a.bootstrap_reps)
    for result in report['effects'].values():
        lo, hi = result['ci95']
        result['equivalent_within_margin'] = bool(
            lo is not None and lo > -a.equivalence_margin and hi < a.equivalence_margin)
    if a.out: V.dump_json(a.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
