#!/usr/bin/env python3
"""FROZEN confirmatory benchmark.  Nothing here is tuned after seeing results.

Fixes what the exploratory pass could not: it reports exact-match (not just similarity),
uses payload alphabets that include the confusable characters the exploratory corpus
deliberately removed (I/O/0/1/l), holds line length constant so output burden is not a
variable, stores every raw response, and measures binding separately from decoding by
reporting which row was actually returned.

Freeze manifest is written next to the results; change any of it and the version bumps.
"""
import os, sys, json, re, math, random, hashlib, subprocess, difflib, time, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_lib import bitmap, canvas, draw_lines, tokens
from pack import exact_fit
import providers as P
import provenance as V

ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = 'confirm-v1'
LINE_CHARS = 56                      # held constant: output burden is not a variable here

# ---- frozen prompts -------------------------------------------------------
# The QUESTION is byte-identical across providers; only the image-delivery sentence
# differs, because codex attaches the image (-i) while claude -p is given a path and
# fetches it with the Read tool.  That asymmetry is unavoidable and is recorded in the
# manifest so comparisons can state it.
HEAD_PATH   = "Read the image at {p}\n\n"
HEAD_ATTACH = "The image attached to this message contains dense small text.\n\n"

def head(provider, path):
    return HEAD_ATTACH if provider == 'codex' else HEAD_PATH.format(p=path)

P_DECODE = ("{head}"
            "Transcribe the FIRST line of text exactly as printed, including the leading\n"
            "4-digit line number. It is at the very top of the image; no row counting is\n"
            "needed. Output STRICT JSON only: {{\"line\": \"...\"}}\n"
            "If the glyphs are unresolvable use {{\"line\": \"UNREADABLE\"}}. Do not guess.\n"
            "Answer from the image alone; do not run any command to crop, zoom or enhance it.")
P_BIND   = ("{head}"
            "Transcribe line {n:04d} exactly as printed, including its leading 4-digit line\n"
            "number. Output STRICT JSON only: {{\"line\": \"...\"}}\n"
            "If the glyphs are unresolvable use {{\"line\": \"UNREADABLE\"}}. Do not guess.\n"
            "Answer from the image alone; do not run any command to crop, zoom or enhance it.")

# ---- frozen payload alphabets --------------------------------------------
ALPHABETS = {
    'alnum_easy': "ABCDEFGHJKLMNPQRSTUVWXYZ23456789",   # confusables removed (optimistic)
    'hex':        "0123456789abcdef",                    # contains 0/1
    'b64':        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",  # I/l/O/0
    'crock32':    "0123456789ABCDEFGHJKMNPQRSTVWXYZ",   # Crockford base32: no I, L, O, U
}
WORDS = ("the of and to in a is that it for as with was on be by not this from at or an "
         "which have has had but they were been their said each she do how if will up "
         "other about out many then them these so some her would make like him into time")

def payload_line(kind, rng, i, width):
    head = f"{i+1:04d} "
    body = width - len(head)
    if kind == 'prose':
        out = []
        while sum(len(w) + 1 for w in out) < body:
            out.append(rng.choice(WORDS.split()))
        return (head + ' '.join(out))[:width]
    a = ALPHABETS[kind]
    groups = []
    while sum(len(g) + 1 for g in groups) < body:
        groups.append(''.join(rng.choice(a) for _ in range(4)))
    return (head + ' '.join(groups))[:width]

CELLS = [('clR5x10', 5, 10), ('6x13', 6, 13), ('8x16', 8, 16)]
PAYLOADS = ['prose', 'alnum_easy', 'hex', 'b64']

def render(cellname, cw, lh, kind, seed):
    uw = (28 * cw) // math.gcd(28, cw)
    w = uw * max(1, round(LINE_CHARS * cw / uw))
    uh = (28 * lh) // math.gcd(28, lh)
    h = min(1456, uh * max(1, round(112 * lh / uh)))
    f = bitmap(cellname)
    cols, rows = w // cw, h // lh
    rng = random.Random(f'{VERSION}|{cellname}|{kind}|{seed}')
    lines = [payload_line(kind, rng, i, cols) for i in range(rows)]
    im, d = canvas(w, h)
    draw_lines(d, f, lines, 0, 0, lh=lh)
    return im, lines, w, h, cols, rows

def cer(a, b):
    if not b: return 1.0
    sm = difflib.SequenceMatcher(None, a, b)
    return 1 - sum(bl.size for bl in sm.get_matching_blocks()) / max(len(a), len(b))

def call(prompt, model, image=None, effort='low', timeout=900):
    prov = P.provider_for(model)
    r = P.run(prompt, model=model, images=[image] if (image and prov == 'codex') else None,
              effort=effort, timeout=timeout, cwd=ROOT)
    ans = P.parse_json_answer(r['text'])
    line = ans.get('line') if isinstance(ans, dict) else None
    return line, V.response_record(r)

def one(job, model, effort='low'):
    cellname, cw, lh, kind, rep = job
    seed = f'rep{rep}'
    im, lines, w, h, cols, rows = render(cellname, cw, lh, kind, seed)
    d = os.path.join(ROOT, 'images', 'CONFIRM'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f'{cellname}_{kind}_{seed}.png'); im.save(p)
    ct, gt = tokens(w, h)
    base = dict(version=VERSION, model=model, provider=P.provider_for(model),
                effort=P.effective_effort(model, effort), cell=f'{cw}x{lh}', font=cellname, payload=kind,
                rep=rep, w=w, h=h, cols=cols, rows=rows, line_chars=cols,
                claude_tokens=ct, chars=cols * rows,
                ch_per_token=round(cols * rows / ct, 2))
    out = []
    t0 = time.time()
    prov = P.provider_for(model)
    got, raw = call(P_DECODE.format(head=head(prov, p)), model, image=p, effort=effort)
    want = lines[0]
    out.append(dict(base, probe='decode', target_row=1, got=got, want=want,
                    image_sha256=V.sha256_file(p), response=raw,
                    exact=(got or '').strip() == want.strip(),
                    cer=round(cer((got or '').strip(), want.strip()), 4),
                    abstain=(got or '').strip().upper() == 'UNREADABLE',
                    seconds=round(time.time() - t0)))
    n = rows // 2
    t0 = time.time()
    got, raw = call(P_BIND.format(head=head(prov, p), n=n), model, image=p, effort=effort)
    want = lines[n - 1]
    hit = None
    if got and got.strip().upper() != 'UNREADABLE':
        best = max(range(len(lines)), key=lambda i: difflib.SequenceMatcher(None, got.strip(), lines[i]).ratio())
        if difflib.SequenceMatcher(None, got.strip(), lines[best]).ratio() > 0.6: hit = best + 1
    out.append(dict(base, probe='bind', target_row=n, got=got, want=want,
                    image_sha256=V.sha256_file(p), response=raw,
                    exact=(got or '').strip() == want.strip(),
                    cer=round(cer((got or '').strip(), want.strip()), 4),
                    abstain=(got or '').strip().upper() == 'UNREADABLE',
                    matched_row=hit, displacement=(hit - n) if hit else None,
                    seconds=round(time.time() - t0)))
    for r in out:
        print(f"  {r['font']:8s} {r['payload']:10s} rep{rep} {r['probe']:6s} "
              f"exact={str(r['exact']):5s} cer={r['cer']:.3f} "
              f"disp={r.get('displacement')} {r['seconds']}s")
    return out

def execute_jobs(jobs, model, effort, workers=3):
    """Execute the frozen matrix while preserving the manifest's declared effort."""
    with ThreadPoolExecutor(workers) as ex:
        return [r for rows in ex.map(lambda job: one(job, model, effort), jobs) for r in rows]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--cells', nargs='*', default=None,
                    help='override cell set (extension run; prompts/grader stay frozen)')
    ap.add_argument('--payloads', nargs='*', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--effort', default='low')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--allow-mutable-model-alias', action='store_true',
                    help='exploratory compatibility only; confirmatory provenance requires an exact id')
    a = ap.parse_args()
    if P.model_is_mutable_alias(a.model) and not a.allow_mutable_model_alias:
        ap.error('--model must be an exact immutable model id (or explicitly opt into an exploratory alias)')
    fn = f'results_{VERSION}{a.tag}_{a.model}.json'
    result_path = os.path.join(ROOT, fn)
    V.require_new_output(result_path, a.overwrite)
    cells = CELLS
    if a.cells:
        cells = [(c, int(re.search(r'(\d+)x', c).group(1)), int(re.search(r'x(\d+)', c).group(1)))
                 for c in a.cells]
    payloads = a.payloads or PAYLOADS
    jobs = [(c, cw, lh, k, r) for (c, cw, lh) in cells for k in payloads
            for r in range(1, a.reps + 1)]
    src = open(__file__).read()
    prov = P.provider_for(a.model)
    manifest = V.manifest(
        experiment=VERSION, model=a.model, provider=prov, effort=a.effort,
        cli_version=P.cli_version(prov), harness_path=__file__,
        prompts={'decode_template': P_DECODE, 'bind_template': P_BIND},
        version=VERSION, reps=a.reps, cells=[c[0] for c in cells],
        payloads=payloads, line_chars=LINE_CHARS,
        image_delivery='attached (-i)' if prov == 'codex' else 'path + Read tool',
        tools_allowed='read-only' if prov == 'codex' else 'Read',
        mutable_model_alias=P.model_is_mutable_alias(a.model), n_runs=len(jobs) * 2)
    print(json.dumps(manifest, indent=1))
    res = execute_jobs(jobs, a.model, a.effort, a.jobs)
    V.dump_json(result_path,
                dict(schema_version=V.RESULT_SCHEMA_VERSION, manifest=manifest, results=res))
    print(f"\n-> {fn}")
