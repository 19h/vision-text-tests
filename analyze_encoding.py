#!/usr/bin/env python3
"""Paired hex/base32 difference intervals and declared-margin equivalence decision."""
import argparse
import collections
import json
import random

import provenance as V
from scipy.stats import beta


def quantile(values, p):
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo = int(pos); hi = min(len(xs) - 1, lo + 1); frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def clopper_pearson(count, n, tail_alpha):
    """One-cell exact interval with ``tail_alpha`` in each tail."""
    if not n: return (None, None)
    lower = 0.0 if count == 0 else float(beta.ppf(tail_alpha, count, n - count + 1))
    upper = 1.0 if count == n else float(beta.ppf(1 - tail_alpha, count + 1, n - count))
    return lower, upper


def paired_difference_interval(diffs, alpha=.05):
    """Conservative 1-alpha CI for P(+1)-P(-1) in paired binary data.

    Bonferroni-simultaneous Clopper-Pearson bounds on the two discordant-cell
    probabilities remain non-degenerate when every pair is concordant, unlike a
    percentile bootstrap.
    """
    n = len(diffs)
    if not n: return [None, None]
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    # Each cell gets a (1-alpha/2) two-sided interval: alpha/4 in each tail.
    lpos, upos = clopper_pearson(pos, n, alpha / 4)
    lneg, uneg = clopper_pearson(neg, n, alpha / 4)
    return [lpos - uneg, upos - lneg]


def compare(rows, bits, margin=.10, reps=10000):
    pairs = collections.defaultdict(dict)
    for row in rows:
        if row.get('bits') == bits:
            pairs[row['rep']][row['enc']] = 1.0 if row.get('value') else 0.0
    diffs = [v['hex'] - v['b32'] for v in pairs.values() if {'hex', 'b32'} <= set(v)]
    if not diffs:
        return {'n': 0, 'difference': None, 'ci95': [None, None],
                'margin': margin, 'equivalent': False}
    rng = random.Random(0xE0C0DE + bits)
    boot = [sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(reps)]
    bootstrap_ci = [quantile(boot, .025), quantile(boot, .975)]
    ci = paired_difference_interval(diffs)
    return {'n': len(diffs), 'difference': sum(diffs) / len(diffs), 'ci95': ci,
            'interval_method': 'bonferroni_clopper_pearson_discordant_cells',
            'bootstrap_ci95_diagnostic': bootstrap_ci,
            'margin': margin, 'equivalent': ci[0] > -margin and ci[1] < margin,
            'discordant_pairs': sum(d != 0 for d in diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('--margin', type=float, default=.10)
    ap.add_argument('--bootstrap-reps', type=int, default=10000)
    ap.add_argument('--out')
    a = ap.parse_args()
    rows = V.load_result_rows(a.result)
    report = {'schema_version': 'encoding-equivalence-analysis-v1',
              'source_sha256': V.sha256_file(a.result),
              'comparisons': {str(bits): compare(rows, bits, a.margin, a.bootstrap_reps)
                              for bits in (64, 128)}}
    if a.out: V.dump_json(a.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
