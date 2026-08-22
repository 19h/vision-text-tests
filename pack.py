#!/usr/bin/env python3
"""Maximum-density packing.

tokens = ceil(w/28) * ceil(h/28).  If w and h are multiples of 28, tokens = w*h/784
exactly, so for a monospace cell cw x lh:

    chars = (w/cw) * (h/lh)          chars/token = 784 / (cw*lh)

which is independent of the page's aspect ratio and of its total size.  Density is
therefore decided entirely by the glyph cell area - proportions only matter for
(a) leaving zero remainder pixels and (b) how many rows the model must count.
"""
import os, re, sys, json, math, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G
from gen_lib import *
from gen_lib import image_tokens, chars_per_token

LCM = 224                                  # lcm(28, 32): divides evenly for both models
def suffix_for(patch):
    return '' if patch == CLAUDE_PATCH else f'_p{patch}'

def token_multiplier(patch):
    return 1.2 if patch == GPT_PATCH else 1.0

def lcm(a, b): return a * b // math.gcd(a, b)

def waste(cw, lh, w, h):
    return 1 - ((w // cw) * cw * (h // lh) * lh) / (w * h)

def exact_fit(cw, lh, lo=600_000, hi=1_050_000, target_rows=112, patch=CLAUDE_PATCH):
    """Zero-waste canvas: 224 | w,h and cw|w, lh|h.  Among the fits, prefer one whose
    row count is near `target_rows`, so candidates differ in glyph size but not in how
    many rows the model has to count.  Falls back to minimum-waste if no exact fit."""
    for unit in (LCM, patch):             # 224 divides both grids; `patch` is provider-exact
        uw, uh = lcm(unit, cw), lcm(unit, lh)
        best = None
        for w in range(uw, 1569, uw):
            for h in range(uh, 1569, uh):
                if not (lo <= w * h <= hi): continue
                rows = h // lh
                score = (abs(rows - target_rows), abs(w / h - 1.0))
                if best is None or score < best[0]: best = (score, w, h)
        if best: return best[1], best[2]
    # no zero-waste canvas in range (e.g. lh=9 never divides a multiple of 224 under 1568px)
    best = None
    for w in range(LCM, 1569, LCM):
        for h in range(LCM, 1569, LCM):
            if not (lo <= w * h <= hi): continue
            rows = h // lh
            score = (round(waste(cw, lh, w, h), 4), abs(rows - target_rows))
            if best is None or score < best[0]: best = (score, w, h)
    return (best[1], best[2]) if best else (896, 896)

def frontier(maxarea=140, patch=CLAUDE_PATCH, mult=1.0):
    names = [f[:-7] for f in os.listdir(X11)
             if re.fullmatch(r'(cl[RBI]\d+x\d+|\d+x\d+[BO]?)\.pcf\.gz', f)]
    out = []
    for n in sorted(names):
        m = re.search(r'(\d+)x(\d+)', n); cw, lh = int(m.group(1)), int(m.group(2))
        if cw * lh > maxarea: continue
        w, h = exact_fit(cw, lh, patch=patch)
        if not w: continue
        out.append(dict(font=n, cw=cw, lh=lh, area=cw * lh, w=w, h=h,
                        cols=w // cw, rows=h // lh, chars=(w // cw) * (h // lh),
                        claude_tokens=w * h // 784,
                        gpt_tokens=math.ceil(w/32) * math.ceil(h/32) * 1.2,
                        tokens=image_tokens(w, h, patch, mult),
                        ch_per_claude=chars_per_token(cw, lh, 28, 1.0),
                        ch_per_gpt=chars_per_token(cw, lh, 32, 1.2),
                        ch_per_token=chars_per_token(cw, lh, patch, mult),
                        waste_pct=round(waste(cw, lh, w, h) * 100, 2)))
    out.sort(key=lambda r: -r['ch_per_token'])
    return out

def packed_page(fontname, w, h, seed, pad=0):
    """Zero-margin maximally packed page of numbered lines."""
    f = bitmap(fontname)
    cols, rows = (w - 2 * pad) // f.cw, (h - 2 * pad) // f.lh
    rng = random.Random(seed)
    lines, meta = numbered_lines(rng, cols, rows)
    im, d = canvas(w, h)
    draw_lines(d, f, lines, pad, pad)
    return im, lines, meta, rows, cols, f

# candidates spanning the legibility floor found in series A (5x8 fails, 6x9 reads)
CANDIDATES = ['clR5x6', 'clR6x6', 'clR5x8', 'clR6x8', 'clR5x10', 'clR7x8',
              '6x9', 'clB6x10', 'clR8x8']

def series_H(patch=CLAUDE_PATCH):
    print("\n[H] packing frontier - zero-waste canvases around the legibility floor")
    for name in CANDIDATES:
        m = re.search(r'(\d+)x(\d+)', name); cw, lh = int(m.group(1)), int(m.group(2))
        w, h = exact_fit(cw, lh, patch=patch)
        im, lines, meta, rows, cols, f = packed_page(name, w, h, 'H' + name)
        G.emit(f"H_packing{suffix_for(patch)}/H_{name}_{w}x{h}.png", im, '\n'.join(lines),
               dict(series='H_packing', style='zero-waste maximal packing', font=f'X11 {name}',
                    cell=f'{cw}x{lh}', cap_height_px=lh, antialiased=False, cols=cols, rows=rows,
                    theoretical_ch_per_token=round(chars_per_token(
                        cw, lh, patch, token_multiplier(patch)), 2), grid_patch=patch),
               G.line_probes(lines, meta, rows))

def series_I():
    """Same cell, same token budget, different page shapes: does row count hurt lookup?"""
    print("\n[I] aspect-ratio probe - identical density, different row counts")
    cw, lh = 6, 9
    for w, h in [(1344, 672), (672, 1344), (896, 1008)]:
        if w % 6 or h % 9:
            h -= h % 9; w -= w % 6
        im, lines, meta, rows, cols, f = packed_page('6x9', w, h, f'I{w}x{h}')
        G.emit(f"I_aspect/I_6x9_{w}x{h}_{rows}rows.png", im, '\n'.join(lines),
               dict(series='I_aspect', style=f'{rows} rows x {cols} cols, same cell',
                    font='X11 6x9', cell='6x9', antialiased=False, cols=cols, rows=rows),
               G.line_probes(lines, meta, rows))

def merge_key():
    p = os.path.join(ROOT, 'ANSWER_KEY.json')
    k = json.load(open(p))
    k['images'].update(G.ENTRIES)
    G.write_catalog(k['images'], generated_by='generate.py + pack.py')
    print(f"answer key updated: {len(k['images'])} images")

# ---------------------------------------------------------------- J: pitch vs glyph size
# Series A-H confound glyph design with row pitch: every X11 face has a fixed cell.
# Here the SAME glyphs are drawn at a larger line pitch, so the two are separable.
PITCH_GRID = [('4x6', 9), ('4x6', 10), ('4x6', 12), ('clR5x6', 9),
              ('clR5x8', 9), ('clR5x8', 10), ('clR6x8', 9), ('5x7', 9)]

def series_J(patch=CLAUDE_PATCH):
    print("\n[J] pitch decoupled from glyph size - same font, taller line step")
    for name, pitch in PITCH_GRID:
        m = re.search(r'(\d+)x(\d+)', name); cw, nat = int(m.group(1)), int(m.group(2))
        w, h = exact_fit(cw, pitch, patch=patch)
        f = bitmap(name)
        cols, rows = w // cw, h // pitch
        rng = random.Random(f'J{name}@{pitch}')
        lines, meta = numbered_lines(rng, cols, rows)
        im, d = canvas(w, h)
        draw_lines(d, f, lines, 0, 0, lh=pitch)          # <- pitch overrides the font cell
        G.emit(f"J_pitch{suffix_for(patch)}/J_{name}_pitch{pitch}_{w}x{h}.png", im, '\n'.join(lines),
               dict(series='J_pitch', style=f'{name} glyphs at {pitch}px row pitch',
                    font=f'X11 {name}', cell=f'{cw}x{pitch}', glyph_cell=f'{cw}x{nat}',
                    pitch=pitch, antialiased=False, cols=cols, rows=rows,
                    theoretical_ch_per_token=round(chars_per_token(
                        cw, pitch, patch, token_multiplier(patch)), 2), grid_patch=patch),
               G.line_probes(lines, meta, rows))

# ---------------------------------------------------------------- K: payload entropy
# Series A-J all render text from a small closed template (17 subjects x 15 verbs x 15
# goods x ...).  A model that resolves a few lines can infer the generator and reconstruct
# the rest by pattern-matching, so those legibility scores are an OPTIMISTIC bound: they
# measure reading-with-priors, not glyph resolution.  These pages carry unguessable
# payload at the same geometry, so the gap between them is the redundancy credit.
def entropy_lines(rng, cols, rows, group=4):
    lines, meta = [], {}
    body = cols - 5
    ngroups = (body + 1) // (group + 1)
    for i in range(rows):
        gs = [''.join(rng.choice(ALNUM) for _ in range(group)) for _ in range(ngroups)]
        lines.append((f"{i+1:04d} " + ' '.join(gs))[:cols])
    meta['passphrase'] = lines[rows // 2][5:5 + group * 2 + 1]
    meta['passphrase_line'] = rows // 2 + 1
    meta['needle_word'] = 'n/a'; meta['needle_count'] = 0; meta['needle_lines'] = []
    return lines, meta

ENTROPY_GRID = [('clR6x8', 8), ('6x9', 9), ('4x6', 9), ('clR5x8', 9), ('6x10', 10), ('6x13', 13),
                ('7x14', 14), ('8x16', 16), ('9x18', 18), ('10x20', 20), ('12x24', 24)]

def series_K(patch=CLAUDE_PATCH):
    print("\n[K] high-entropy payload - same geometry, no linguistic redundancy")
    for name, pitch in ENTROPY_GRID:
        m = re.search(r'(\d+)x(\d+)', name); cw = int(m.group(1))
        w, h = exact_fit(cw, pitch, patch=patch)
        f = bitmap(name)
        cols, rows = w // cw, h // pitch
        rng = random.Random(f'K{name}@{pitch}')
        lines, meta = entropy_lines(rng, cols, rows)
        im, d = canvas(w, h)
        draw_lines(d, f, lines, 0, 0, lh=pitch)
        probes = [dict(id='verbatim_first', q="Transcribe the first line exactly.", a=lines[0]),
                  dict(id='verbatim_second', q="Transcribe the second line exactly.", a=lines[1]),
                  dict(id='verbatim_last', q="Transcribe the final line exactly.", a=lines[-1]),
                  dict(id='n_lines', q="How many lines are in the image?", a=str(rows))]
        G.emit(f"K_entropy{suffix_for(patch)}/K_{name}_pitch{pitch}_{w}x{h}.png", im, '\n'.join(lines),
               dict(series='K_entropy', style=f'random 4-char groups, {pitch}px pitch',
                    font=f'X11 {name}', cell=f'{cw}x{pitch}', pitch=pitch, antialiased=False,
                    cols=cols, rows=rows, payload='high-entropy',
                    theoretical_ch_per_token=round(chars_per_token(
                        cw, pitch, patch, token_multiplier(patch)), 2), grid_patch=patch), probes)

# K2: high-entropy payload on NARROW pages.  K's 224-char lines make transcribing five of
# them a ~660-character random-string task, so a K failure could be output stamina or
# refusal rather than reading.  Same cells, ~56-char lines: same density, small answer.
K2_GRID = [('6x9', 9), ('6x10', 10), ('6x13', 13), ('8x16', 16), ('9x18', 18)]

def series_K2(patch=CLAUDE_PATCH):
    print("\n[K2] high-entropy, narrow pages - reading isolated from transcription burden")
    for name, pitch in K2_GRID:
        cw = int(re.search(r'(\d+)x', name).group(1))
        uw = lcm(patch, cw)
        w = uw * max(1, round(56 * cw / uw))          # ~56 columns
        uh = lcm(patch, pitch)
        max_h = 1600 if patch == 32 else 1456         # provider max square
        target_h = uh * max(1, round(112 * pitch / uh))
        # Preserve the original Claude corpus's 1456px cap even for the one cell
        # (9x18) where that leaves a partial final row.  The provider-native p32
        # series was explicitly designed to remain grid/cell aligned.
        h = (uh * max(1, min(round(112 * pitch / uh), max_h // uh))
             if patch == GPT_PATCH else min(max_h, target_h))
        f = bitmap(name)
        cols, rows = w // cw, h // pitch
        rng = random.Random(f'K2{name}')
        lines, meta = entropy_lines(rng, cols, rows)
        im, d = canvas(w, h)
        draw_lines(d, f, lines, 0, 0, lh=pitch)
        probes = [dict(id='verbatim_first', q="Transcribe the first line exactly.", a=lines[0]),
                  dict(id='verbatim_second', q="Transcribe the second line exactly.", a=lines[1]),
                  dict(id='verbatim_last', q="Transcribe the final line exactly.", a=lines[-1])]
        G.emit(f"K2_entropy_narrow{suffix_for(patch)}/K2_{name}_{w}x{h}.png", im, '\n'.join(lines),
               dict(series='K2_entropy_narrow', style=f'random 4-char groups, {cols}-char lines',
                    font=f'X11 {name}', cell=f'{cw}x{pitch}', pitch=pitch, antialiased=False,
                    cols=cols, rows=rows, payload='high-entropy-narrow',
                    theoretical_ch_per_token=round(chars_per_token(
                        cw, pitch, patch, token_multiplier(patch)), 2), grid_patch=patch), probes)

def build_series(patch, include_aspect=False):
    series_H(patch)
    if include_aspect:
        series_I()
    series_J(patch)
    series_K(patch)
    series_K2(patch)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--patch', type=int, default=None,
                    help='build one grid only; default rebuilds the complete 28+32 grid corpus')
    ap.add_argument('--mult', type=float, default=None)
    a = ap.parse_args()
    patch = a.patch or CLAUDE_PATCH
    mult = a.mult if a.mult is not None else token_multiplier(patch)
    if a.table:
        print(f"{'font':9s}{'cell':>6s}{'px/ch':>6s}{'canvas':>12s}{'cols':>5s}{'rows':>5s}"
              f"{'chars':>8s}{'cl-tok':>7s}{'ch/cl':>7s}{'ch/gpt':>7s}{'waste':>7s}")
        for r in frontier(patch=patch, mult=mult):
            print(f"{r['font']:9s}{r['cw']}x{r['lh']:<4d}{r['area']:>6d}"
                  f"{str(r['w'])+'x'+str(r['h']):>12s}{r['cols']:>5d}{r['rows']:>5d}"
                  f"{r['chars']:>8d}{r['claude_tokens']:>7d}{r['ch_per_claude']:>7.1f}"
                  f"{r['ch_per_gpt']:>7.1f}{r['waste_pct']:>6.1f}%")
        return
    if a.patch is None:
        build_series(CLAUDE_PATCH, include_aspect=True)
        build_series(GPT_PATCH, include_aspect=False)
    else:
        build_series(a.patch, include_aspect=(a.patch == CLAUDE_PATCH))
    merge_key()

if __name__ == '__main__':
    main()
