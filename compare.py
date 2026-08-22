#!/usr/bin/env python3
"""Cross-provider comparison over whatever result files exist.

Every table here is paired on IDENTICAL images. The one thing that cannot be matched is
image delivery: codex attaches images with -i, claude -p is given a path and uses the Read
tool. Stimuli identical, delivery not - stated on every table rather than buried.
"""
import json, glob, os, re, sys, statistics as st
import provenance as V
ROOT = os.path.dirname(os.path.abspath(__file__))
def _load(p):
    try: return V.result_rows(json.load(open(os.path.join(ROOT, p))))
    except Exception: return None
def _blob(p):
    try: return json.load(open(os.path.join(ROOT, p)))
    except Exception: return None
def lb(k, n):
    try:
        from scipy.stats import beta
        return beta.ppf(0.05, k, n - k + 1) if k > 0 else 0.0
    except ImportError: return float('nan')
def fisher(k1, n1, k2, n2):
    try:
        from scipy.stats import fisher_exact
        return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
    except ImportError: return float('nan')
NN = lambda s: re.sub(r'\s+', '', str(s))

def ladder():
    files = {'opus': 'results_opus.json', 'sonnet': 'results_sonnet.json'}
    for f in sorted(glob.glob(os.path.join(ROOT, 'results_ladder_*.json'))):
        files[os.path.basename(f)[15:-5]] = os.path.basename(f)
    got = {m: {r['glyph']: r for r in _load(p) or []} for m, p in files.items()}
    got = {m: v for m, v in got.items() if v}
    if not got: return
    print("\n=== BITMAP LADDER (identical images) ===")
    print(f"{'model':16s}{'mean lookup':>13s}{'mean legible':>14s}{'s/image':>9s}{'n':>4s}")
    for m, d in got.items():
        rs = [r for r in d.values() if r.get('code_acc') is not None]
        if not rs: continue
        print(f"{m:16s}{sum(r['code_acc'] for r in rs)/len(rs)*100:12.1f}%"
              f"{sum(r.get('legibility') or 0 for r in rs)/len(rs):>14.2f}"
              f"{sum(r['seconds'] for r in rs)/len(rs):>9.0f}{len(rs):>4d}")

def legibility():
    files = {'opus': 'results_edge_opus.json'}
    for f in sorted(glob.glob(os.path.join(ROOT, 'results_edge_gpt-5.6-*.json'))):
        files[os.path.basename(f)[13:-5]] = os.path.basename(f)
    # key on the FILE, not (cell, rate): the sol-native _p32 pages can collide with
    # 28-grid pages on that tuple and silently overwrite them. Pairing must be on
    # byte-identical stimuli.
    got = {m: {r['file']: r['legibility'] for r in _load(p) or []} for m, p in files.items()}
    got = {m: v for m, v in got.items() if v}
    base = got.get('opus')
    if not base or len(got) < 2: return
    print("\n=== LEGIBILITY ONLY, paired vs opus (first/last lines, no row counting) ===")
    for m, d in got.items():
        if m == 'opus': continue
        pairs = [(base[k], d[k]) for k in base if k in d]
        if not pairs: continue
        diffs = [b - a for a, b in pairs]
        try:
            from scipy.stats import wilcoxon
            nz = [x for x in diffs if abs(x) > 1e-9]
            p = wilcoxon(nz).pvalue if len(nz) > 5 else float('nan')
        except Exception: p = float('nan')
        print(f"  {m:16s} n={len(pairs):3d}  opus {sum(a for a,_ in pairs)/len(pairs):.3f} -> "
              f"{m.split('-')[-1]} {sum(b for _,b in pairs)/len(pairs):.3f}   "
              f"delta {sum(diffs)/len(diffs):+.3f}   Wilcoxon p={p:.4f}")

def spans():
    got = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'results_span_6x13_*.json'))):
        b = os.path.basename(f)[len('results_span_6x13_'):-5]
        alpha, model = b.split('_', 1)
        got[(model, alpha)] = _load(os.path.basename(f))
    if not got: return
    print("\n=== SHORT SPANS n=20 (literal = reading + length compliance; prefix = reading) ===")
    print(f"{'model':16s}{'alpha':9s}{'metric':9s}" + ''.join(f"{'sp'+str(n):>8s}" for n in (8,16,32,51)) + f"{'total':>9s}{'95% LB':>9s}")
    for (model, alpha), d in sorted(got.items()):
        for label, key in (('literal', lambda r: r['exact']), ('prefix', lambda r: r.get('prefix', r['exact']))):
            cells, tot, totn = [], 0, 0
            for n in (8, 16, 32, 51):
                rs = [r for r in d if r['span'] == n]
                k = sum(key(r) for r in rs); tot += k; totn += len(rs)
                cells.append(f"{k}/{len(rs)}" if rs else '-')
            print(f"{model:16s}{alpha:9s}{label:9s}" + ''.join(f"{c:>8s}" for c in cells) +
                  f"{f'{tot}/{totn}':>9s}{lb(tot,totn):>9.3f}")

def delim():
    got = {os.path.basename(f)[len('results_delim_6x13_'):-5]: _load(os.path.basename(f))
           for f in sorted(glob.glob(os.path.join(ROOT, 'results_delim_6x13_*.json')))}
    got = {k: v for k, v in got.items() if v and 'UNPAIRED' not in k}
    if not got: return
    print("\n=== DELIMITED EQUAL-INFORMATION FIELDS n=20 (decoded value equality) ===")
    print(f"{'model':16s}{'bits':>6s}{'enc':>6s}{'chars':>7s}{'literal':>10s}{'value':>10s}{'95% LB':>9s}")
    for m, d in sorted(got.items()):
        for b in (64, 128):
            for e in ('hex', 'b32'):
                rs = [r for r in d if r['bits'] == b and r['enc'] == e]
                if not rs: continue
                v = sum(r['value'] for r in rs)
                lit = f"{sum(r['literal'] for r in rs)}/{len(rs)}"
                print(f"{m:16s}{b:>6d}{e:>6s}{rs[0]['chars']:>7d}"
                      f"{lit:>10s}"
                      f"{f'{v}/{len(rs)}':>10s}{lb(v,len(rs)):>9.3f}")

def confirm():
    files = {'opus': ['results_confirm-v1_opus.json', 'results_confirm-v1-ext_opus.json']}
    for f in sorted(glob.glob(os.path.join(ROOT, 'results_confirm-v1_gpt-5.6-*.json'))):
        m = json.load(open(f))['manifest']['model']
        files[m] = [os.path.basename(f)]
        x = os.path.basename(f).replace('confirm-v1_', 'confirm-v1-ext_')
        if os.path.exists(os.path.join(ROOT, x)): files[m].append(x)
    BASE = {'clR5x10', '6x13', '8x16'}
    print("\n=== WHOLE 56-CHAR RECORDS, high-entropy payload, base matrix ===")
    print(f"{'model':16s}{'exact':>9s}{'rate':>7s}{'95% LB':>9s}{'bind disp=0':>13s}")
    for m, ps in files.items():
        R = [r for p in ps for r in (_load(p) or [])]
        hi = [r for r in R if r['probe'] == 'decode' and r['payload'] != 'prose' and r['font'] in BASE]
        bd = [r for r in R if r['probe'] == 'bind' and r['font'] in BASE]
        if not hi: continue
        k = sum(r['exact'] for r in hi)
        d0 = sum(1 for r in bd if r.get('displacement') == 0)
        print(f"{m:16s}{f'{k}/{len(hi)}':>9s}{k/len(hi):>7.2f}{lb(k,len(hi)):>9.3f}"
              f"{f'{d0}/{len(bd)}':>13s}")

if __name__ == '__main__':
    print("Claude vs GPT-5.6 - identical images; image DELIVERY differs (path+Read vs attached -i)")
    errors = []
    for fn in (ladder, legibility, confirm, spans, delim):
        try: fn()
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
            print(f"  [{fn.__name__}: {e}]")
    if errors:
        raise SystemExit("comparison failures: " + "; ".join(errors))
