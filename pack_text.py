#!/usr/bin/env python3
"""Pack arbitrary text into the fewest image tokens.

  python3 pack_text.py notes.txt --out packed/
  python3 pack_text.py notes.txt --font clR5x6 --numbers --out packed/

Density is 784/(cw*lh) chars per Claude image token and is independent of page
proportions, so the only thing that matters is choosing the smallest glyph cell the
model can still read, then leaving zero remainder pixels.
"""
import os, sys, math, argparse, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_lib import bitmap, canvas, draw_lines, tokens, CLAUDE_PATCH
from pack import exact_fit, waste
import re

# Measured operating points (Opus, legibility-only probe, see README).
# name: (font, pitch, max_cols, measured legibility, note)
PRESETS = {
    'prose': ('clR5x10', 10, None, 0.99, 'redundant text: 15.7 ch/token, legibility 0.99'),
    'dense': ('clR5x8',   9,  None, 0.85, 'push for prose: 17.4 ch/token, legibility 0.85'),
    'data':  ('6x13',    13,  64,   0.95, 'unguessable payload: 9.7 ch/token, needs short lines'),
    'safe':  ('8x16',    16,  64,   0.98, 'anything, including hashes: 6.1 ch/token'),
}

def wrap_all(text, cols, reflow, numbers):
    """Text -> lines of at most `cols` chars.  reflow=True ignores the input's own line
    breaks and fills every row, which is what makes the theoretical density real."""
    pre = 7 if numbers else 0
    body = cols - pre
    if body < 8: return None
    if reflow:
        stream = ' '.join(text.split())
        lines = textwrap.wrap(stream, body) or ['']
    else:
        lines = []
        for raw in text.replace('\t', '    ').split('\n'):
            if not raw.strip(): lines.append(''); continue
            lines += textwrap.wrap(raw, body) or ['']
    if numbers:
        lines = [f"{i+1:>5d}  {l}" for i, l in enumerate(lines)]
    return lines

def optimise(text, fontname, max_px=1568, numbers=False, reflow=True, unit=None,
             max_cols=None, tol=1.15):
    """Pick the canvas that renders this text in the fewest image tokens.

    Density per *filled* cell is fixed at 784/(cw*lh), but ragged text leaves cells
    empty, and the fill rate depends on how the column count matches the text's line
    lengths.  So the page shape does matter for real documents - just not for synthetic
    full-width text.  This searches the legal canvases and minimises total tokens."""
    m = re.search(r'(\d+)x(\d+)', fontname)
    cw, lh = int(m.group(1)), int(m.group(2))
    cands = []
    for u in ([unit] if unit else [224, CLAUDE_PATCH]):   # 224 aligns both grids; 28 is Claude-only
        uw, uh = (u * cw) // math.gcd(u, cw), (u * lh) // math.gcd(u, lh)
        for w in range(uw, max_px + 1, uw):
            if max_cols and w // cw > max_cols: break     # long lines wreck high-entropy reads
            lines = wrap_all(text, w // cw, reflow, numbers)
            if not lines: continue
            ink = sum(len(l) for l in lines)
            for h in range(uh, max_px + 1, uh):
                rows = h // lh
                pages = math.ceil(len(lines) / rows)
                tok = pages * math.ceil(w / CLAUDE_PATCH) * math.ceil(h / CLAUDE_PATCH)
                fill = ink / max(1, pages * rows * (w // cw))
                cands.append((tok, pages, -fill, u != 224, w, h, lines, ink))
    # both alignments are collected: 224 steps are coarse and often miss a much better fit
    if not cands: raise SystemExit('nothing fits - text too wide for this font')
    # cheapest first, but do not accept 70 tiny images to save 3% of tokens:
    # among everything within 5% of the best token count, take the fewest, fullest pages.
    floor_tok = min(c[0] for c in cands)
    near = [c for c in cands if c[0] <= floor_tok * tol]
    best = min(near, key=lambda c: (c[1], c[2], c[3], c[0]))   # pages, fill, grid-alignment, tokens
    tok, _pages, _fill, _u28, w, h, lines, ink = best
    cols, rows = w // cw, h // lh
    return lines, dict(font=fontname, cell=f'{cw}x{lh}', w=w, h=h, cols=cols, rows=rows,
                       px_per_char=cw * lh, pages=math.ceil(len(lines) / rows),
                       claude_tokens=tok, ink_chars=ink,
                       fill_pct=round(100 * ink / max(1, math.ceil(len(lines)/rows) * rows * cols), 1),
                       waste_pct=round(waste(cw, lh, w, h) * 100, 2))

def build(text, fontname, max_px=1568, numbers=False, reflow=True, max_cols=None, tol=1.15):
    lines, geo = optimise(text, fontname, max_px, numbers, reflow, max_cols=max_cols, tol=tol)
    f = bitmap(fontname)
    pages = [lines[i:i + geo['rows']] for i in range(0, len(lines), geo['rows'])] or [[]]
    out = []
    for pg in pages:
        im, d = canvas(geo['w'], geo['h'])
        draw_lines(d, f, pg, 0, 0)
        out.append((im, '\n'.join(pg)))
    return out, geo

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('--out', default='packed')
    ap.add_argument('--font', default=None)
    ap.add_argument('--preset', default='prose', choices=list(PRESETS))
    ap.add_argument('--max-cols', type=int, default=None,
                    help='cap line length; high-entropy payloads need <=64 (measured)')
    ap.add_argument('--max-px', type=int, default=1568)
    ap.add_argument('--numbers', action='store_true', help='prefix line numbers (helps addressing)')
    ap.add_argument('--keep-lines', action='store_true',
                    help='preserve the input line breaks (lower fill, lower density)')
    a = ap.parse_args()
    font = a.font or PRESETS[a.preset][0]
    max_cols = a.max_cols if a.max_cols is not None else PRESETS[a.preset][2]
    text = open(a.input).read()
    pages, geo = build(text, font, a.max_px, a.numbers, reflow=not a.keep_lines,
                       max_cols=max_cols)
    os.makedirs(a.out, exist_ok=True)
    tot_tok = tot_ch = 0
    for i, (im, txt) in enumerate(pages, 1):
        p = os.path.join(a.out, f"page{i:03d}.png"); im.save(p)
        ct, gt = tokens(im.width, im.height)
        tot_tok += ct; tot_ch += len(txt.replace('\n', ''))
        print(f"  {p}  {im.width}x{im.height}  {len(txt.splitlines())} lines  {ct} claude-tok")
    # measured on this corpus with `claude -p --output-format json` (see README):
    # template prose 2.17 chars/text-token, random 4-char groups 1.14.  4.0 is typical English.
    ANCHORS = [('typical English', 4.0), ('mixed prose+codes', 2.17), ('random codes', 1.14)]
    text_tok = round(len(text) / 4)
    print(f"\n{geo['font']} ({geo['cell']}, {geo['px_per_char']} px/char)  "
          f"{geo['w']}x{geo['h']} = {geo['cols']} cols x {geo['rows']} rows, "
          f"{geo['fill_pct']}% of cells filled")
    print(f"{len(pages)} page(s), {tot_ch:,} chars, {tot_tok:,} image tokens "
          f"= {tot_ch/max(tot_tok,1):.1f} chars/token")
    print("versus sending the same characters as text:")
    for lbl, cpt in ANCHORS:
        tt = round(tot_ch / cpt)
        print(f"  at {cpt:>4} chars/token ({lbl:18s}) = {tt:8,} text tokens -> "
              f"{tt/max(tot_tok,1):5.1f}x {'cheaper as image' if tt > tot_tok else 'cheaper as text'}")
    json.dump(dict(geo=geo, pages=len(pages), chars=tot_ch, claude_tokens=tot_tok,
                   text_tokens_equiv=text_tok, compression=round(text_tok / max(tot_tok, 1), 2)),
              open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
