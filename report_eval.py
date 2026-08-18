#!/usr/bin/env python3
"""Merge results_*.json into a comparison table."""
import json, glob, os, collections, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
rows = collections.defaultdict(dict)
models, order = [], []
for f in sorted(glob.glob(os.path.join(ROOT, 'results_*.json'))):
    for r in json.load(open(f)):
        m = r['model']
        if m not in models: models.append(m)
        if r['file'] not in order: order.append(r['file'])
        rows[r['file']][m] = r
order.sort(key=lambda k: rows[k][models[0]]['ch_per_tok'], reverse=True)

def cell(r):
    if r is None: return '  --  '
    if r.get('code_acc') is None: return ' FAIL '
    ab = r.get('abstained', 0)
    return f"{r['code_acc']*100:3.0f}%{'*' if ab else ' '}"

hdr = f"| {'glyph':>6s} | {'ch/tok':>6s} | " + ' | '.join(f"{m} codes | {m} verbatim" for m in models) + " |"
sep = '|' + '---|' * (2 + 2 * len(models))
print(hdr); print(sep)
for k in order:
    r0 = rows[k][models[0]]
    line = f"| {r0['glyph']:>6s} | {r0['ch_per_tok']:6.1f} | "
    for m in models:
        r = rows[k].get(m)
        line += cell(r) + ' | '
        line += (f"{r['verbatim_sim']:.2f}" if r and r.get('verbatim_sim') is not None else '--') + ' | '
    print(line)
print("\n* = model abstained on at least one probe (said UNREADABLE)")
for m in models:
    rs = [rows[k][m] for k in order if m in rows[k] and rows[k][m].get('code_acc') is not None]
    if not rs: continue
    perfect = [r for r in rs if r['code_acc'] == 1.0 and (r['verbatim_sim'] or 0) > 0.95]
    print(f"{m}: {len(rs)} images, mean code acc {sum(r['code_acc'] for r in rs)/len(rs)*100:.0f}%, "
          f"clean floor = {min((r['glyph'] for r in perfect), key=lambda g: int(g.split('x')[0])*int(g.split('x')[1]) if 'x' in g else 999, default='none')}, "
          f"mean {sum(r['seconds'] for r in rs)/len(rs):.0f}s/image")
