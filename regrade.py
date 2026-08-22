#!/usr/bin/env python3
"""Re-apply the grader to stored results (no new API calls; read-only by default)."""
import argparse, json, os, sys, glob
from datetime import datetime, timezone
import provenance as V
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import grade, best_line, norm, KEY, ROOT
SKIP = ('confirm', 'ebind1', 'span', 'delim', 'edge', 'geometry', 'verifier', 'layout')
ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='*')
ap.add_argument('--write', action='store_true', help='persist changes; default is an audit only')
a = ap.parse_args()
for f in a.files or glob.glob(os.path.join(ROOT, 'results_*.json')):
    if not a.files and any(x in os.path.basename(f) for x in SKIP): continue
    before_sha = V.sha256_file(f)
    blob = json.load(open(f))
    data = V.result_rows(blob)
    if not data: continue
    changed = 0
    for r in data:
        if not isinstance(r, dict) or 'file' not in r or r['file'] not in KEY: continue
        v = KEY[r['file']]
        gt = open(os.path.join(ROOT, v['groundtruth'])).read().split('\n')
        codes, sims, legs, ab, mis = [], [], [], 0, 0
        for pid, d in r.get('details', {}).items():
            verdict, sc = grade(pid, d['got'], d['want'], gt)
            if verdict != d['verdict']: changed += 1
            d['verdict'], d['score'] = verdict, round(sc, 3)
            if verdict == 'abstain': ab += 1
            if verdict.startswith('misaddr'): mis += 1
            if pid.startswith('code_'): codes.append(sc)
            elif pid.startswith('verbatim'):
                sims.append(sc)
                legs.append(0.0 if d['got'].upper() == 'UNREADABLE' else best_line(norm(d['got']), gt)[1])
        if codes: r['code_acc'] = round(sum(codes)/len(codes), 3)
        if sims:  r['verbatim_sim'] = round(sum(sims)/len(sims), 3)
        if legs:  r['legibility'] = round(sum(legs)/len(legs), 3)
        r['abstained'], r['misaddressed'] = ab, mis
    if a.write:
        if isinstance(blob, dict) and 'results' in blob:
            blob['results'] = data
            blob.setdefault('regrade_history', []).append({
                'created_at': datetime.now(timezone.utc).isoformat(),
                'grader_sha256': V.source_sha(__file__), 'input_sha256': before_sha,
                'verdicts_changed': changed,
            })
            V.dump_json(f, blob)
        else:
            V.dump_json(f, data)
            audit_path = os.path.join(ROOT, 'regrade_audit_' + os.path.basename(f))
            V.dump_json(audit_path, {
                'schema_version': 'regrade-audit-v1',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'grader_sha256': V.source_sha(__file__), 'input_sha256': before_sha,
                'output_sha256': V.sha256_file(f), 'verdicts_changed': changed,
            })
    mode = 'written' if a.write else 'audit only; not written'
    print(f"{os.path.basename(f)}: {len(data)} results, {changed} verdicts changed ({mode})")
