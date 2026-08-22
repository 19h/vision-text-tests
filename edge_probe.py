#!/usr/bin/env python3
"""Legibility-only probe: transcribe the first/last lines, which need no row counting.

Separates 'cannot resolve the glyphs' from 'cannot find row 0421' and from the model
simply declining under a no-guessing instruction.
"""
import argparse, json, os, re, sys, subprocess, difflib, time
from concurrent.futures import ThreadPoolExecutor
import providers as P
import provenance as V
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
    if not isinstance(a, dict):
        return dict(file=k, model=model, provider=prov,
                    effort=P.effective_effort(model, effort), ok=False,
                    note=str(rr['text'])[:120], image_sha256=V.sha256_file(img),
                    response=V.response_record(rr), seconds=round(time.time()-t0))
    got = [x for x in (a.get('first') or [])][:3] + [x for x in (a.get('last') or [])][:2]
    want = gt[:3] + gt[-2:]
    scores, abst = [], 0
    for g, w in zip(got, want):
        if str(g).strip().upper() == 'UNREADABLE': abst += 1; scores.append(0.0)
        else: scores.append(sim(str(g), w))
    res = dict(file=k, model=model, provider=P.provider_for(model),
               effort=P.effective_effort(model, effort), cell=v.get('cell'), ch_per_tok=v['chars_per_claude_token'],
               ch_per_tok_gpt=v.get('chars_per_gpt_token'),
               legibility=round(sum(scores)/max(1,len(scores)), 3), abstained=abst,
               confidence=a.get('confidence'), per_line=[round(s,2) for s in scores],
               seconds=round(time.time()-t0), sample_got=str(got[0])[:100] if got else '',
               sample_want=want[0][:100], image_sha256=V.sha256_file(img),
               response=V.response_record(rr))
    print(f"  {v.get('cell'):>5s} {v['chars_per_claude_token']:5.1f} ch/tok  legible {res['legibility']:.2f}"
          f"  abstain {abst}/5  conf={res['confidence']}  {res['seconds']}s")
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('patterns', nargs='*')
    ap.add_argument('--model', required=True)
    ap.add_argument('--effort', default='low')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--tag', default='', help='result suffix for an independent run')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--allow-mutable-model-alias', action='store_true')
    a = ap.parse_args()
    model, effort, pats = a.model, a.effort, a.patterns
    if P.model_is_mutable_alias(model) and not a.allow_mutable_model_alias:
        ap.error('--model must be an exact immutable model id (or opt into an exploratory alias)')
    p = os.path.join(ROOT, f'results_edge{a.tag}_{model}.json')
    V.require_new_output(p, a.overwrite)
    ks = [k for k in KEY if any(p in k for p in pats)]
    print(f"{len(ks)} images, model={model}")
    with ThreadPoolExecutor(a.jobs) as ex:
        out = list(ex.map(lambda k: run(k, model, effort=effort), ks))
    prov = P.provider_for(model)
    manifest = V.manifest(
        experiment='legibility-edge-v2', model=model, provider=prov, effort=effort,
        cli_version=P.cli_version(prov), harness_path=__file__, prompts={'edge': PROMPT},
        patterns=pats, replacement_semantics='one immutable artifact per tagged run')
    V.dump_json(p, dict(schema_version=V.RESULT_SCHEMA_VERSION,
                        manifest=manifest, results=out)); print('->', p)
