#!/usr/bin/env python3
"""Re-apply the grader to stored results (no new API calls)."""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import grade, best_line, norm, KEY, ROOT
for f in sys.argv[1:] or glob.glob(os.path.join(ROOT, 'results_*.json')):
    data = json.load(open(f)); changed = 0
    for r in data:
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
    json.dump(data, open(f, 'w'), indent=1)
    print(f"{os.path.basename(f)}: {len(data)} results, {changed} verdicts changed")
