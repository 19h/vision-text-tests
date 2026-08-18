#!/usr/bin/env python3
"""P(exact) as a function of requested span length, at fixed image density.

The confirmatory pass showed CER plateauing near 0.02-0.04 regardless of glyph size, so
exact recovery of a 56-character record fails at every cell.  If the residual is a roughly
per-character error rate rather than a resolution limit, short anchored spans should be
recoverable exactly even where whole lines are not.  This measures that directly.
"""
import os, sys, json, re, random, subprocess, time, argparse, difflib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confirm import render, cer, VERSION
ROOT = os.path.dirname(os.path.abspath(__file__))

PROMPT = ("Read the image at {p}\n\n"
          "Look at the FIRST line of text, at the very top of the image. After its 4-digit\n"
          "line number and a space, transcribe the NEXT {n} characters exactly, spaces\n"
          "included. Output STRICT JSON only: {{\"span\": \"...\"}}\n"
          "If unresolvable use {{\"span\": \"UNREADABLE\"}}. Do not guess.\n"
          "Answer from the image alone; do not run any command to crop, zoom or enhance it.")

def one(job, model):
    cell, cw, lh, kind, n, rep = job
    im, lines, w, h, cols, rows = render(cell, cw, lh, kind, f'span{rep}')
    d = os.path.join(ROOT, 'images', 'SPAN'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f'{cell}_{kind}_r{rep}.png'); im.save(p)
    want = lines[0][5:5 + n]
    t0 = time.time()
    r = subprocess.run(['claude', '-p', '--allowedTools', 'Read', '--disallowedTools',
                        'Bash,Write,Edit,Glob,Grep,Task,WebFetch,WebSearch',
                        '--model', model, PROMPT.format(p=p, n=n)],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    got = ''
    for c in reversed(re.findall(r'\{.*?\}', r.stdout, re.S)):
        try: got = json.loads(c).get('span', ''); break
        except json.JSONDecodeError: continue
    got = (got or '').strip()
    res = dict(cell=f'{cw}x{lh}', font=cell, payload=kind, span=n, rep=rep,
               exact=got == want, cer=round(cer(got, want), 4), got=got, want=want,
               abstain=got.upper() == 'UNREADABLE', seconds=round(time.time() - t0))
    print(f"  {cell:7s} {kind:4s} span={n:3d} rep{rep} exact={str(res['exact']):5s} "
          f"cer={res['cer']:.3f} {res['seconds']}s")
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='opus'); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--cell', default='9x18'); ap.add_argument('--payload', default='hex')
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    cw, lh = (int(x) for x in re.match(r'(?:\D*)(\d+)x(\d+)', a.cell).groups())
    jobs = [(a.cell, cw, lh, a.payload, n, r) for n in (8, 16, 32, 51) for r in range(1, a.reps + 1)]
    print(f"{len(jobs)} runs: {a.cell} {a.payload}, spans 8/16/32/51, {a.reps} reps")
    with ThreadPoolExecutor(a.jobs) as ex:
        out = list(ex.map(lambda j: one(j, a.model), jobs))
    json.dump(out, open(os.path.join(ROOT, f'results_span_{a.cell}_{a.payload}_{a.model}.json'), 'w'), indent=1)
    import statistics as st
    print("\nspan  exact   mean CER")
    for n in (8, 16, 32, 51):
        rs = [r for r in out if r['span'] == n]
        print(f"{n:4d}  {sum(r['exact'] for r in rs)}/{len(rs)}   {st.mean(r['cer'] for r in rs):.3f}")
