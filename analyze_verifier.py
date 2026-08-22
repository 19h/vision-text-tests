#!/usr/bin/env python3
"""Cluster-aware safety analysis for verifier_probe.py results."""
from __future__ import annotations

import argparse
import json

from analyze_ebind import cluster_interval
import provenance as V


def event_interval(rows, value, reps):
    derived = [dict(r, outcome='EVENT' if bool(r.get(value)) else 'OTHER') for r in rows]
    return cluster_interval(derived, 'EVENT', reps=reps, seed=0x5AFE)


def analyze(rows, reps=5000):
    correct = [r for r in rows if r.get('truth_matches') is True]
    wrong = [r for r in rows if r.get('truth_matches') is False]
    present_wrong = [r for r in wrong if r.get('candidate_kind') == 'wrong_valid_decoy']
    absent_wrong = [r for r in wrong if r.get('candidate_kind') == 'no_answer_candidate']
    by_item = {}
    for row in rows:
        by_item.setdefault(row['item_id'], {})[row['candidate_kind']] = row
    discrimination = []
    for item, candidates in by_item.items():
        if {'correct', 'wrong_valid_decoy'} <= set(candidates):
            good, bad = candidates['correct'], candidates['wrong_valid_decoy']
            discrimination.append({
                'item_id': item, 'sampling_page': good.get('sampling_page'),
                'outcome': 'PASS' if (good['runtime_accept'] and not bad['runtime_accept']) else 'FAIL',
            })
    return {
        'true_accept': event_interval(correct, 'runtime_accept', reps),
        'false_accept_all_wrong': event_interval(wrong, 'runtime_accept', reps),
        'false_accept_present_decoy': event_interval(present_wrong, 'runtime_accept', reps),
        'false_accept_no_answer': event_interval(absent_wrong, 'runtime_accept', reps),
        'paired_discrimination': cluster_interval(discrimination, 'PASS', reps=reps,
                                                  seed=0xD15C) if discrimination else None,
        'parse_failures': sum(not r.get('parse_ok') for r in rows),
    }


def markdown(report):
    lines = ['# E-BIND semantic-verifier analysis', '',
             'The verifier is fail closed. Parse failures count as rejects, never accepts.', '',
             '| endpoint | rate | 95% interval | items | calls | clustering |',
             '|---|---:|---:|---:|---:|---|']
    for name in ('true_accept', 'false_accept_all_wrong', 'false_accept_present_decoy',
                 'false_accept_no_answer', 'paired_discrimination'):
        value = report.get(name)
        if not value: continue
        lines.append(f"| {name} | {value['rate']:.3f} | "
                     f"[{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}] | "
                     f"{value['n_items']} | {value['n_calls']} | {value['cluster']} |")
    lines += ['', f"Parse failures: {report['parse_failures']}", '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('--bootstrap-reps', type=int, default=5000)
    ap.add_argument('--json-out')
    ap.add_argument('--markdown-out')
    a = ap.parse_args()
    rows = V.load_result_rows(a.result)
    report = {'schema_version': 'verifier-analysis-v1',
              'source': {'path': a.result, 'sha256': V.sha256_file(a.result)}}
    report.update(analyze(rows, a.bootstrap_reps))
    if a.json_out: V.dump_json(a.json_out, report)
    text = markdown(report)
    if a.markdown_out:
        with open(a.markdown_out, 'w') as f: f.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
