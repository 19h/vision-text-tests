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
import providers as P
import provenance as V
ROOT = os.path.dirname(os.path.abspath(__file__))
B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"     # Crockford

PROMPT = ("{head}"
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
def _strip_separators(s):
    return re.sub(r'[\s\-]', '', str(s))

def dec_hex(s):
    t = _strip_separators(s)
    return int(t, 16) if t and re.fullmatch(r'[0-9a-fA-F]+', t) else None
def dec_b32(s):
    s = _strip_separators(s).upper().replace('O','0').replace('I','1').replace('L','1')
    v = 0
    for ch in s:
        if ch not in B32: return None
        v = v * 32 + B32.index(ch)
    return v

def decode_value(s, enc, bits):
    """Protocol decoder: permitted separators/aliases, exact length/range, canonical re-encode."""
    t = _strip_separators(s)
    if enc == 'hex':
        if len(t) != bits // 4 or not re.fullmatch(r'[0-9a-fA-F]+', t): return None
        return int(t, 16)
    canonical = t.upper().replace('O','0').replace('I','1').replace('L','1')
    if len(canonical) != math.ceil(bits / 5): return None
    v = dec_b32(t)
    if v is None or not (0 <= v < (1 << bits)): return None
    return v if enc_b32(v, bits) == canonical else None

def render(cell, cw, lh, enc, bits, seed):
    # seed must NOT include `enc`: hex and base32 must encode the SAME bitstrings
    # so the comparison is paired and McNemar applies.
    rng = random.Random(f'delim|{cell}|{bits}|{seed}')
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

def one(job, model, effort='low'):
    cell, cw, lh, enc, bits, rep = job
    im, lines, vals, w, h, cols, rows = render(cell, cw, lh, enc, bits, f'r{rep}')
    dd = os.path.join(ROOT, 'images', 'DELIM'); os.makedirs(dd, exist_ok=True)
    p = os.path.join(dd, f'{cell}_{enc}{bits}_r{rep}.png'); im.save(p)
    want_s = lines[0][lines[0].index('[') + 1:lines[0].index(']')]
    t0 = time.time()
    prov = P.provider_for(model)
    rr = P.run(PROMPT.format(head=P.image_head(prov, p)), model=model,
               images=P.images_for(prov, [p]), effort=effort, timeout=900, cwd=ROOT)
    ans = P.parse_json_answer(rr['text'])
    got = (ans or {}).get('field', '') if isinstance(ans, dict) else ''
    got = (got or '').strip()
    gv = None if got.upper() == 'UNREADABLE' else decode_value(got, enc, bits)
    wv = vals[0]
    res = dict(cell=f'{cw}x{lh}', enc=enc, bits=bits, chars=len(want_s), rep=rep,
               model=model, provider=P.provider_for(model),
               effort=P.effective_effort(model, effort),
               literal=got == want_s, value=(gv == wv), got=got, want=want_s,
               abstain=got.upper() == 'UNREADABLE',
               cer=round(1 - difflib.SequenceMatcher(None, got, want_s).ratio(), 4),
               ch_per_token=round(cols * rows / tokens(w, h)[0], 2),
               image_sha256=V.sha256_file(p), response=V.response_record(rr),
               seconds=round(time.time() - t0))
    print(f"  {enc:6s} {bits:3d}b/{len(want_s):2d}ch rep{rep:<3d} literal={str(res['literal']):5s} "
          f"value={str(res['value']):5s} {res['seconds']}s")
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--cell', default='6x13'); ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--effort', default='low')
    ap.add_argument('--tag', default='', help='suffix for independent campaigns (for example _effort-medium)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--allow-mutable-model-alias', action='store_true')
    a = ap.parse_args()
    if P.model_is_mutable_alias(a.model) and not a.allow_mutable_model_alias:
        ap.error('--model must be an exact immutable model id (or opt into an exploratory alias)')
    output_path = os.path.join(ROOT, f'results_delim_{a.cell}{a.tag}_{a.model}.json')
    V.require_new_output(output_path, a.overwrite)
    cw, lh = (int(x) for x in re.match(r'(?:\D*)(\d+)x(\d+)', a.cell).groups())
    jobs = [(a.cell, cw, lh, e, b, r) for b in (128, 64) for e in ('hex', 'b32')
            for r in range(1, a.reps + 1)]
    print(f"{len(jobs)} runs: {a.cell}, equal-information delimited fields")
    with ThreadPoolExecutor(a.jobs) as ex:
        out = list(ex.map(lambda j: one(j, a.model, a.effort), jobs))
    prov = P.provider_for(a.model)
    manifest = V.manifest(
        experiment='delimited-equal-information-v2', model=a.model, provider=prov,
        effort=a.effort, cli_version=P.cli_version(prov), harness_path=__file__,
        prompts={'field': PROMPT}, cell=a.cell, reps=a.reps,
        bits=[64, 128], encodings=['hex', 'b32'])
    V.dump_json(output_path,
                dict(schema_version=V.RESULT_SCHEMA_VERSION, manifest=manifest, results=out))
    print(f"\n{'enc':6s}{'bits':>5s}{'chars':>6s}{'literal':>9s}{'value':>8s}")
    for b in (128, 64):
        for e in ('hex', 'b32'):
            rs = [r for r in out if r['bits'] == b and r['enc'] == e]
            print(f"{e:6s}{b:>5d}{rs[0]['chars']:>6d}{sum(r['literal'] for r in rs):>6d}/{len(rs):<3d}"
                  f"{sum(r['value'] for r in rs):>5d}/{len(rs):<3d}")
