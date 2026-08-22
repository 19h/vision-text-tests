#!/usr/bin/env python3
"""Run the benchmark against `claude -p` and grade the answers automatically.

  python3 run_eval.py --model sonnet --series A_bitmap_ladder
  python3 run_eval.py --model opus --images A01 A05 A09 --jobs 4
"""
import argparse, json, os, re, subprocess, sys, time, difflib
from concurrent.futures import ThreadPoolExecutor

import providers as P
ROOT = os.path.dirname(os.path.abspath(__file__))
KEY = json.load(open(os.path.join(ROOT, 'ANSWER_KEY.json')))['images']
GRADED = ('n_lines', 'passphrase', 'rare_count', 'verbatim_mid')   # + every code_*

def probes_for(v):
    ps = [p for p in v['probes'] if p['id'].startswith('code_') or p['id'] in GRADED]
    return ps or v['probes'][:5]

def build_prompt(path, ps, provider='claude'):
    qs = '\n'.join(f'  "{p["id"]}": {p["q"]}' for p in ps)
    return (P.image_head(provider, path, 'is the page referred to below')
          + "It contains dense small text. Answer the questions below from the image.\n"
            "Output STRICT JSON and nothing else: one key per question id, string values.\n"
            'If you genuinely cannot read what a question asks for, answer "UNREADABLE".\n'
            "Do not guess or invent plausible-looking text - the content is random, so a\n"
            "guess is always wrong and scores worse than admitting it is illegible.\n"
            "Answer purely from looking at the image. Do NOT run any command, script or\n"
            "image-processing tool to crop, zoom or enhance it - that invalidates the test.\n\n"
            f"Questions:\n{qs}\n")

def call(path, ps, model, timeout, effort='low'):
    t0 = time.time()
    prov = P.provider_for(model)
    try:
        r = P.run(build_prompt(path, ps, prov), model=model,
                  images=P.images_for(prov, [path]), effort=effort, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0, 'timeout'
    ans = P.parse_json_answer(r['text'])
    if not isinstance(ans, dict):
        return None, time.time() - t0, str(r['text'])[:200]
    return ans, time.time() - t0, None

def norm(s): return re.sub(r'\s+', ' ', str(s)).strip()

def best_line(g, gt):
    """Index and similarity of the ground-truth line the answer matches best."""
    best, bs = 0, 0.0
    for i, l in enumerate(gt):
        r = difflib.SequenceMatcher(None, g, norm(l)[:max(len(g), 40)]).ratio()
        if r > bs: best, bs = i, r
    return best, bs

def grade(pid, got, want, gt=None):
    """gt = ground-truth lines, used to tell a mis-addressed answer from a misread one."""
    g, w = norm(got), norm(want)
    if g.upper() in ('UNREADABLE', 'ILLEGIBLE', ''): return 'abstain', 0.0
    if pid.startswith('code_'):
        if g.upper() == w.upper(): return 'ok', 1.0
        if gt:                      # did it read a real code from a different line?
            hits = [i + 1 for i, l in enumerate(gt) if len(l) > 10 and l[5:10].upper() == g.upper()]
            if hits: return f'misaddressed(line {hits[0]} not {int(pid.split("_")[1])})', 0.0
        return 'wrong', 0.0
    if pid == 'passphrase':
        pw = w.split(' (')[0]
        return ('ok', 1.0) if pw.upper() in g.upper() else ('wrong', 0.0)
    if pid in ('n_lines', 'rare_count'):
        gi = re.search(r'-?\d+', g); wi = re.search(r'-?\d+', w)
        if not gi: return 'wrong', 0.0
        return ('ok', 1.0) if gi.group() == wi.group() else ('wrong', 0.0)
    if pid.startswith('verbatim'):
        sim = difflib.SequenceMatcher(None, g, w).ratio()
        if sim > 0.98: return 'ok', sim
        if gt:   # did it transcribe a DIFFERENT line correctly?  addressing, not legibility
            bi, bs = best_line(g, gt)
            if bs > 0.9 and norm(gt[bi]) != w:
                return f'misaddressed(line {bi + 1})', sim
        return ('partial' if sim > 0.5 else 'wrong'), sim
    return ('ok', 1.0) if g == w else ('wrong', 0.0)

def run_one(item, model, timeout, effort='low'):
    k, v = item
    ps = probes_for(v)
    gt = open(os.path.join(ROOT, v['groundtruth'])).read().split('\n')
    ans, secs, err = call(os.path.join(ROOT, 'images', k), ps, model, timeout, effort)
    res = dict(file=k, model=model, provider=P.provider_for(model), effort=effort,
               seconds=round(secs, 1), error=err,
               glyph=v.get('cell') or f"{v.get('size_px')}px", chars=v['chars'],
               ch_per_tok=v['chars_per_claude_token'], details={})
    if ans is None:
        res.update(code_acc=None, verbatim_sim=None, score=None)
        print(f"  {model:7s} {k:44s} FAILED ({err})")
        return res
    codes, sims, legs, absts = [], [], [], 0
    for p in ps:
        verdict, sc = grade(p['id'], ans.get(p['id'], ''), p['a'], gt)
        res['details'][p['id']] = dict(got=norm(ans.get(p['id'], ''))[:400],
                                       want=norm(p['a'])[:400], verdict=verdict, score=round(sc, 3))
        if verdict == 'abstain': absts += 1
        if verdict.startswith('misaddr'): res['misaddressed'] = res.get('misaddressed', 0) + 1
        if p['id'].startswith('code_'): codes.append(sc)
        elif p['id'].startswith('verbatim'):
            sims.append(sc)
            g = norm(ans.get(p['id'], ''))
            legs.append(best_line(g, gt)[1] if g.upper() != 'UNREADABLE' else 0.0)
    res.update(code_acc=round(sum(codes) / len(codes), 3) if codes else None,
               verbatim_sim=round(sum(sims) / len(sims), 3) if sims else None,
               legibility=round(sum(legs) / len(legs), 3) if legs else None,
               abstained=absts,
               n_lines=res['details'].get('n_lines', {}).get('verdict'),
               passphrase=res['details'].get('passphrase', {}).get('verdict'))
    print(f"  {model:7s} {k:44s} codes {res['code_acc']:.2f}  verbatim {res['verbatim_sim'] or 0:.2f}"
          f"  legible {res['legibility'] or 0:.2f}"
          f"  pass={res['passphrase']}  lines={res['n_lines']}  abstain={absts}  {secs:.0f}s")
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sonnet')
    ap.add_argument('--series', nargs='*', default=[])
    ap.add_argument('--images', nargs='*', default=[])
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--out', default='results.json')
    ap.add_argument('--effort', default='low')
    a = ap.parse_args()
    items = [(k, v) for k, v in KEY.items()
             if (not a.series or v['series'] in a.series)
             and (not a.images or any(s.lower() in k.lower() for s in a.images))]
    print(f"{len(items)} images x model={a.model}, {a.jobs} parallel")
    with ThreadPoolExecutor(a.jobs) as ex:
        out = list(ex.map(lambda it: run_one(it, a.model, a.timeout, a.effort), items))
    p = os.path.join(ROOT, a.out)
    prev = json.load(open(p)) if os.path.exists(p) else []
    json.dump(prev + out, open(p, 'w'), indent=1)
    print(f"-> {p}")
