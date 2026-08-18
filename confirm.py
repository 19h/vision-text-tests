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

ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = 'confirm-v1'
LINE_CHARS = 56                      # held constant: output burden is not a variable here

# ---- frozen prompts -------------------------------------------------------
P_DECODE = ("Read the image at {p}\n\n"
            "Transcribe the FIRST line of text exactly as printed, including the leading\n"
            "4-digit line number. It is at the very top of the image; no row counting is\n"
            "needed. Output STRICT JSON only: {{\"line\": \"...\"}}\n"
            "If the glyphs are unresolvable use {{\"line\": \"UNREADABLE\"}}. Do not guess.\n"
            "Answer from the image alone; do not run any command to crop, zoom or enhance it.")
P_BIND   = ("Read the image at {p}\n\n"
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

def call(prompt, model, timeout=600):
    r = subprocess.run(['claude', '-p', '--allowedTools', 'Read', '--disallowedTools',
                        'Bash,Write,Edit,Glob,Grep,Task,WebFetch,WebSearch,NotebookEdit',
                        '--model', model, prompt],
                       capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    m = re.findall(r'\{.*?\}', r.stdout, re.S)
    for cand in reversed(m):
        try: return json.loads(cand).get('line', ''), r.stdout
        except json.JSONDecodeError: continue
    return None, r.stdout

def one(job, model):
    cellname, cw, lh, kind, rep = job
    seed = f'rep{rep}'
    im, lines, w, h, cols, rows = render(cellname, cw, lh, kind, seed)
    d = os.path.join(ROOT, 'images', 'CONFIRM'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f'{cellname}_{kind}_{seed}.png'); im.save(p)
    ct, gt = tokens(w, h)
    base = dict(version=VERSION, model=model, cell=f'{cw}x{lh}', font=cellname, payload=kind,
                rep=rep, w=w, h=h, cols=cols, rows=rows, line_chars=cols,
                claude_tokens=ct, chars=cols * rows,
                ch_per_token=round(cols * rows / ct, 2))
    out = []
    t0 = time.time()
    got, raw = call(P_DECODE.format(p=p), model)
    want = lines[0]
    out.append(dict(base, probe='decode', target_row=1, got=got, want=want, raw=raw[-2000:],
                    exact=(got or '').strip() == want.strip(),
                    cer=round(cer((got or '').strip(), want.strip()), 4),
                    abstain=(got or '').strip().upper() == 'UNREADABLE',
                    seconds=round(time.time() - t0)))
    n = rows // 2
    t0 = time.time()
    got, raw = call(P_BIND.format(p=p, n=n), model)
    want = lines[n - 1]
    hit = None
    if got and got.strip().upper() != 'UNREADABLE':
        best = max(range(len(lines)), key=lambda i: difflib.SequenceMatcher(None, got.strip(), lines[i]).ratio())
        if difflib.SequenceMatcher(None, got.strip(), lines[best]).ratio() > 0.6: hit = best + 1
    out.append(dict(base, probe='bind', target_row=n, got=got, want=want, raw=raw[-2000:],
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='opus'); ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--cells', nargs='*', default=None,
                    help='override cell set (extension run; prompts/grader stay frozen)')
    ap.add_argument('--payloads', nargs='*', default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    cells = CELLS
    if a.cells:
        cells = [(c, int(re.search(r'(\d+)x', c).group(1)), int(re.search(r'x(\d+)', c).group(1)))
                 for c in a.cells]
    payloads = a.payloads or PAYLOADS
    jobs = [(c, cw, lh, k, r) for (c, cw, lh) in cells for k in payloads
            for r in range(1, a.reps + 1)]
    src = open(__file__).read()
    manifest = dict(version=VERSION, model=a.model, reps=a.reps, cells=[c[0] for c in CELLS],
                    payloads=PAYLOADS, line_chars=LINE_CHARS,
                    prompt_decode_sha=hashlib.sha256(P_DECODE.encode()).hexdigest()[:16],
                    prompt_bind_sha=hashlib.sha256(P_BIND.encode()).hexdigest()[:16],
                    harness_sha=hashlib.sha256(src.encode()).hexdigest()[:16],
                    cli_version=subprocess.run(['claude','--version'],capture_output=True,text=True).stdout.strip(),
                    tools_allowed='Read', n_runs=len(jobs) * 2)
    print(json.dumps(manifest, indent=1))
    with ThreadPoolExecutor(a.jobs) as ex:
        res = [r for rs in ex.map(lambda j: one(j, a.model), jobs) for r in rs]
    fn = f'results_{VERSION}{a.tag}_{a.model}.json'
    json.dump(dict(manifest=manifest, results=res), open(os.path.join(ROOT, fn), 'w'), indent=1)
    print(f"\n-> {fn}")
