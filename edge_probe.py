#!/usr/bin/env python3
"""Legibility-only probe: transcribe the first/last lines, which need no row counting.

Separates 'cannot resolve the glyphs' from 'cannot find row 0421' and from the model
simply declining under a no-guessing instruction.
"""
import json, os, re, sys, subprocess, difflib, time
from concurrent.futures import ThreadPoolExecutor
import providers as P
ROOT = os.path.dirname(os.path.abspath(__file__))
KEY = json.load(open(os.path.join(ROOT, 'ANSWER_KEY.json')))['images']
PROMPT = ("{head}"
          "Transcribe the FIRST 3 lines and the LAST 2 lines of text, exactly as printed.\n"
          "They are at the very top and very bottom of the image, so you do not need to\n"
          "count rows. Output STRICT JSON only:\n"
          '{{"first": ["line1", "line2", "line3"], "last": ["line1", "line2"],\n'
          '  "confidence": "high|medium|low"}}\n'
          "If the glyphs are genuinely unresolvable, use the string UNREADABLE for that\n"
          "line. Answer from the image alone - do not run any command to crop or zoom it.")

def sim(a, b): return difflib.SequenceMatcher(None, re.sub(r'\s+', ' ', a).strip(),
                                              re.sub(r'\s+', ' ', b).strip()).ratio()

def run(k, model='opus', timeout=900, effort='low'):
    v = KEY[k]
    gt = open(os.path.join(ROOT, v['groundtruth'])).read().split('\n')
    t0 = time.time()
    prov = P.provider_for(model); img = os.path.join(ROOT, 'images', k)
    rr = P.run(PROMPT.format(head=P.image_head(prov, img)), model=model,
               images=P.images_for(prov, [img]), effort=effort, timeout=timeout, cwd=ROOT)
    a = P.parse_json_answer(rr['text'])
    if not isinstance(a, dict): return dict(file=k, ok=False, note=str(rr['text'])[:120])
    got = [x for x in (a.get('first') or [])][:3] + [x for x in (a.get('last') or [])][:2]
    want = gt[:3] + gt[-2:]
    scores, abst = [], 0
    for g, w in zip(got, want):
        if str(g).strip().upper() == 'UNREADABLE': abst += 1; scores.append(0.0)
        else: scores.append(sim(str(g), w))
    res = dict(file=k, model=model, provider=P.provider_for(model), effort=effort, cell=v.get('cell'), ch_per_tok=v['chars_per_claude_token'],
               ch_per_tok_gpt=v.get('chars_per_gpt_token'),
               legibility=round(sum(scores)/max(1,len(scores)), 3), abstained=abst,
               confidence=a.get('confidence'), per_line=[round(s,2) for s in scores],
               seconds=round(time.time()-t0), sample_got=str(got[0])[:100] if got else '',
               sample_want=want[0][:100])
    print(f"  {v.get('cell'):>5s} {v['chars_per_claude_token']:5.1f} ch/tok  legible {res['legibility']:.2f}"
          f"  abstain {abst}/5  conf={res['confidence']}  {res['seconds']}s")
    return res

if __name__ == '__main__':
    pats = [a for a in sys.argv[1:] if not a.startswith('--')]
    model = 'opus'; effort = 'low'
    for a in sys.argv[1:]:
        if a.startswith('--model='): model = a.split('=')[1]
        if a.startswith('--effort='): effort = a.split('=')[1]
    ks = [k for k in KEY if any(p in k for p in pats)]
    print(f"{len(ks)} images, model={model}")
    with ThreadPoolExecutor(3) as ex:
        out = list(ex.map(lambda k: run(k, model, effort=effort), ks))
    p = os.path.join(ROOT, f'results_edge_{model}.json')
    prev = json.load(open(p)) if os.path.exists(p) else []
    prev = [r for r in prev if r['file'] not in {x['file'] for x in out}]   # newest run wins
    json.dump(prev + out, open(p, 'w'), indent=1); print('->', p)
