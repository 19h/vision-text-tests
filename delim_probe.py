#!/usr/bin/env python3
"""Delimited equal-information fields: removes the counting confound, holds bits constant.

The span probe said "the next N characters", which makes the model count to N.  Here the
field is enclosed in visible brackets, so the boundary is in the image.  Hex and base32
encode the SAME underlying bitstring, so 32 hex chars and 26 base32 chars are compared at
equal information (128 bits) rather than equal character count.
Scored on decoded VALUE equality, not literal string equality.
"""
import os, sys, json, re, random, subprocess, time, argparse, difflib, math
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_lib import bitmap, canvas, draw_lines, tokens
ROOT = os.path.dirname(os.path.abspath(__file__))
B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"     # Crockford

PROMPT = ("Read the image at {p}\n\n"
          "The FIRST line of text, at the very top, contains one field enclosed in square\n"
          "brackets. Transcribe exactly the characters BETWEEN the brackets - not the\n"
          "brackets themselves, and nothing else.\n"
          "Output STRICT JSON only: {{\"field\": \"...\"}}\n"
          "If unresolvable use {{\"field\": \"UNREADABLE\"}}. Do not guess.\n"
          "Answer from the image alone; do not run any command to crop, zoom or enhance it.")

def enc_hex(v, bits): return f"%0{bits//4}x" % v
def enc_b32(v, bits):
    n = math.ceil(bits / 5); s = ''
    for _ in range(n): s = B32[v & 31] + s; v >>= 5
    return s
def dec_hex(s): return int(re.sub(r'[^0-9a-fA-F]', '', s), 16) if re.sub(r'[^0-9a-fA-F]','',s) else None
def dec_b32(s):
    s = re.sub(r'[\s\-]', '', s).upper().replace('O','0').replace('I','1').replace('L','1')
    v = 0
    for ch in s:
        if ch not in B32: return None
        v = v * 32 + B32.index(ch)
    return v

def render(cell, cw, lh, enc, bits, seed):
    rng = random.Random(f'delim|{cell}|{enc}|{bits}|{seed}')
    f = bitmap(cell)
    w = 336 if cw == 6 else 448 if cw == 8 else 504
    h = min(1456, (28 * lh // math.gcd(28, lh)) * max(1, round(112 * lh / (28 * lh // math.gcd(28, lh)))))
    cols, rows = w // cw, h // lh
    vals, lines = [], []
    for i in range(rows):
        v = rng.getrandbits(bits)
        s = enc_hex(v, bits) if enc == 'hex' else enc_b32(v, bits)
        vals.append(v)
        lines.append(f"{i+1:04d} [{s}]"[:cols])
    im, d = canvas(w, h); draw_lines(d, f, lines, 0, 0, lh=lh)
    return im, lines, vals, w, h, cols, rows

def one(job, model):
    cell, cw, lh, enc, bits, rep = job
    im, lines, vals, w, h, cols, rows = render(cell, cw, lh, enc, bits, f'r{rep}')
    dd = os.path.join(ROOT, 'images', 'DELIM'); os.makedirs(dd, exist_ok=True)
    p = os.path.join(dd, f'{cell}_{enc}{bits}_r{rep}.png'); im.save(p)
    want_s = lines[0][lines[0].index('[') + 1:lines[0].index(']')]
    t0 = time.time()
    r = subprocess.run(['claude', '-p', '--allowedTools', 'Read', '--disallowedTools',
                        'Bash,Write,Edit,Glob,Grep,Task,WebFetch,WebSearch',
                        '--model', model, PROMPT.format(p=p)],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    got = ''
    for c in reversed(re.findall(r'\{.*?\}', r.stdout, re.S)):
        try: got = json.loads(c).get('field', ''); break
        except json.JSONDecodeError: continue
    got = (got or '').strip()
    dec = dec_hex if enc == 'hex' else dec_b32
    gv, wv = (None if got.upper() == 'UNREADABLE' else dec(got)), vals[0]
    res = dict(cell=f'{cw}x{lh}', enc=enc, bits=bits, chars=len(want_s), rep=rep,
               literal=got == want_s, value=(gv == wv), got=got, want=want_s,
               abstain=got.upper() == 'UNREADABLE',
               cer=round(1 - difflib.SequenceMatcher(None, got, want_s).ratio(), 4),
               ch_per_token=round(cols * rows / tokens(w, h)[0], 2), seconds=round(time.time() - t0))
    print(f"  {enc:6s} {bits:3d}b/{len(want_s):2d}ch rep{rep:<3d} literal={str(res['literal']):5s} "
          f"value={str(res['value']):5s} {res['seconds']}s")
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='opus'); ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--cell', default='6x13'); ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    cw, lh = (int(x) for x in re.match(r'(?:\D*)(\d+)x(\d+)', a.cell).groups())
    jobs = [(a.cell, cw, lh, e, b, r) for b in (128, 64) for e in ('hex', 'b32')
            for r in range(1, a.reps + 1)]
    print(f"{len(jobs)} runs: {a.cell}, equal-information delimited fields")
    with ThreadPoolExecutor(a.jobs) as ex:
        out = list(ex.map(lambda j: one(j, a.model), jobs))
    json.dump(out, open(os.path.join(ROOT, f'results_delim_{a.cell}_{a.model}.json'), 'w'), indent=1)
    print(f"\n{'enc':6s}{'bits':>5s}{'chars':>6s}{'literal':>9s}{'value':>8s}")
    for b in (128, 64):
        for e in ('hex', 'b32'):
            rs = [r for r in out if r['bits'] == b and r['enc'] == e]
            print(f"{e:6s}{b:>5d}{rs[0]['chars']:>6d}{sum(r['literal'] for r in rs):>6d}/{len(rs):<3d}"
                  f"{sum(r['value'] for r in rs):>5d}/{len(rs):<3d}")
