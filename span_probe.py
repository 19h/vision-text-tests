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
import providers as P
ROOT = os.path.dirname(os.path.abspath(__file__))

PROMPT = ("{head}"
          "Look at the FIRST line of text, at the very top of the image. After its 4-digit\n"
          "line number and a space, transcribe the NEXT {n} characters exactly, spaces\n"
          "included. Output STRICT JSON only: {{\"span\": \"...\"}}\n"
          "If unresolvable use {{\"span\": \"UNREADABLE\"}}. Do not guess.\n"
          "Answer from the image alone; do not run any command to crop, zoom or enhance it.")

def one(job, model, effort='low'):
    cell, cw, lh, kind, n, rep = job
    im, lines, w, h, cols, rows = render(cell, cw, lh, kind, f'span{rep}')
    d = os.path.join(ROOT, 'images', 'SPAN'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f'{cell}_{kind}_r{rep}.png'); im.save(p)
    want = lines[0][5:5 + n]
    t0 = time.time()
    prov = P.provider_for(model)
    rr = P.run(PROMPT.format(head=P.image_head(prov, p), n=n), model=model,
               images=P.images_for(prov, [p]), effort=effort, timeout=900, cwd=ROOT)
    ans = P.parse_json_answer(rr['text'])
    got = (ans or {}).get('span', '') if isinstance(ans, dict) else ''
    got = (got or '').strip()
    # `exact` conflates reading with obeying the requested length: a model that reads
    # correctly but completes the visible 4-char group returns N+1 chars and scores 0.
    # `prefix` isolates reading; report both.
    _n = lambda s: re.sub(r'\s+', '', str(s))
    res = dict(cell=f'{cw}x{lh}', font=cell, payload=kind, span=n, rep=rep,
               model=model, provider=P.provider_for(model), effort=effort,
               exact=got == want,
               prefix=bool(_n(want)) and _n(got).startswith(_n(want)),
               overlong=len(_n(got)) > len(_n(want)),
               cer=round(cer(got, want), 4), got=got, want=want,
               abstain=got.upper() == 'UNREADABLE', seconds=round(time.time() - t0))
    print(f"  {cell:7s} {kind:4s} span={n:3d} rep{rep} exact={str(res['exact']):5s} "
          f"cer={res['cer']:.3f} {res['seconds']}s")
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='opus'); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--cell', default='9x18'); ap.add_argument('--payload', default='hex')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--effort', default='low')
    a = ap.parse_args()
    cw, lh = (int(x) for x in re.match(r'(?:\D*)(\d+)x(\d+)', a.cell).groups())
    jobs = [(a.cell, cw, lh, a.payload, n, r) for n in (8, 16, 32, 51) for r in range(1, a.reps + 1)]
    print(f"{len(jobs)} runs: {a.cell} {a.payload}, spans 8/16/32/51, {a.reps} reps")
    with ThreadPoolExecutor(a.jobs) as ex:
        out = list(ex.map(lambda j: one(j, a.model, a.effort), jobs))
    json.dump(out, open(os.path.join(ROOT, f'results_span_{a.cell}_{a.payload}_{a.model}.json'), 'w'), indent=1)
    import statistics as st
    print("\nspan  literal  prefix  overlong  mean CER")
    for n in (8, 16, 32, 51):
        rs = [r for r in out if r['span'] == n]
        if not rs: continue
        print(f"{n:4d}  {sum(r['exact'] for r in rs):>5d}/{len(rs):<3d}{sum(r['prefix'] for r in rs):>5d}/{len(rs):<3d}"
              f"{sum(r.get('overlong', False) for r in rs):>7d}   {st.mean(r['cer'] for r in rs):.3f}")
