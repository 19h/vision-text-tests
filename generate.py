#!/usr/bin/env python3
"""Build the visual-text-comprehension test corpus.  python3 generate.py"""
import os, json, math, random, csv
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from gen_lib import *

ENTRIES = {}

def emit(rel, im, text, info, probes):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    gtrel = rel.replace('/', '__').rsplit('.', 1)[0] + '.txt'
    with open(os.path.join(GT, gtrel), 'w') as f:
        f.write(text)
    e = dict(file=rel, groundtruth='groundtruth/' + gtrel, probes=probes)
    e.update(info); e.update(stats(im.width, im.height, text))
    ENTRIES[rel] = e
    print(f"  {rel:52s} {im.width}x{im.height} {e['chars']:6d} chars "
          f"{e['chars_per_claude_token']:6.2f} ch/claude-tok")

def line_probes(lines, meta, rows):
    idx = sorted({max(0, rows // 12), rows // 3, (rows * 2) // 3, rows - 1})
    p = [dict(id='n_lines', q="How many numbered lines are in this image?", a=str(rows)),
         dict(id='passphrase', q="An exclamation-marked PASSPHRASE appears on one line. What is it, and on which line number?",
              a=f"{meta['passphrase']} (line {meta['passphrase_line']:04d})"),
         dict(id='rare_count', q=f"How many times does the word {meta['needle_word']} appear?",
              a=f"{meta['needle_count']} (lines {meta['needle_lines']})")]
    for i in idx:
        p.append(dict(id=f'code_{i+1:04d}', q=f"What is the 5-character code on line {i+1:04d}?",
                      a=lines[i][5:10]))
    p.append(dict(id='verbatim_mid', q=f"Transcribe line {idx[1]+1:04d} exactly.", a=lines[idx[1]]))
    p.append(dict(id='verbatim_last', q="Transcribe the final line exactly.", a=lines[-1]))
    return p

# ============================================================ A: bitmap ladder
BITMAPS = ['4x6','5x7','5x8','6x9','6x10','6x12','6x13','7x13','7x14','8x13','8x16','9x15','9x18','10x20','12x24']
def series_A(W=896, H=896):
    print("\n[A] pixel-perfect bitmap ladder (no anti-aliasing)")
    for i, name in enumerate(BITMAPS, 1):
        f = bitmap(name)
        cols, rows = fits(f, W, H, pad=4)
        rng = random.Random('A' + name)
        lines, meta = numbered_lines(rng, cols, rows)
        im, d = canvas(W, H)
        draw_lines(d, f, lines, 4, 4)
        txt = '\n'.join(lines)
        emit(f"A_bitmap_ladder/A{i:02d}_{name}.png", im, txt,
             dict(series='A_bitmap_ladder', style='numbered lines, dense',
                  font=f'X11 misc {name} bitmap', cell=name, cap_height_px=int(name.split('x')[1]),
                  antialiased=False, cols=cols, rows=rows), line_probes(lines, meta, rows))

# ============================================================ B: truetype mono ladder
TTF_SIZES = [5,6,7,8,9,10,11,12,14,16,20,24,32]
def series_B(W=896, H=896):
    print("\n[B] anti-aliased monospace ladder (DejaVu Sans Mono)")
    for i, s in enumerate(TTF_SIZES, 1):
        f = ttf(DEJAVU_MONO, s)
        cols, rows = fits(f, W, H, pad=5)
        rng = random.Random('B%d' % s)
        lines, meta = numbered_lines(rng, cols, rows)
        im, d = canvas(W, H)
        draw_lines(d, f, lines, 5, 3)
        emit(f"B_mono_ttf_ladder/B{i:02d}_dejavu_mono_{s:02d}px.png", im, '\n'.join(lines),
             dict(series='B_mono_ttf_ladder', style='numbered lines, dense',
                  font='DejaVu Sans Mono', size_px=s, antialiased=True,
                  cols=cols, rows=rows), line_probes(lines, meta, rows))

def series_B2(W=896, H=896):
    print("\n[B2] same sizes, anti-aliasing OFF (1-bit rendering)")
    for i, s in enumerate([6,8,10,12,16], 1):
        f = ttf(DEJAVU_MONO, s)
        cols, rows = fits(f, W, H, pad=5)
        rng = random.Random('B2%d' % s)
        lines, meta = numbered_lines(rng, cols, rows)
        im, d = canvas(W, H)
        draw_lines(d, f, lines, 5, 3, aa=False)
        emit(f"B_mono_ttf_ladder/B{i:02d}b_dejavu_mono_{s:02d}px_noAA.png", im, '\n'.join(lines),
             dict(series='B_mono_ttf_ladder', style='numbered lines, no anti-aliasing',
                  font='DejaVu Sans Mono', size_px=s, antialiased=False,
                  cols=cols, rows=rows), line_probes(lines, meta, rows))

# ============================================================ C: proportional prose
def series_C(W=896, H=896):
    print("\n[C] proportional prose ladder (DejaVu Sans) - reading comprehension")
    for i, s in enumerate([5,6,7,8,9,10,12,14,16,20,24], 1):
        f = ttf(DEJAVU_SANS, s, mono=False)
        rng = random.Random('C%d' % s)
        rows = (H - 16) // f.lh
        approx = int(rows * (W - 16) / (f.cw * 0.92)) + 400
        body, facts = prose(rng, approx)
        pw = f"{rng.choice(['VIOLET','AMBER','COBALT'])}-{rng.choice(['BADGER','MARLIN','FALCON'])}-{rng.randint(1000,9999)}"
        needle = f" Remember this: the passphrase is {pw}. "
        cut = len(body) // 2
        body = body[:cut] + needle + body[cut:]
        wrapped = wrap_to(f, body, W - 16)[:rows]
        im, d = canvas(W, H)
        draw_lines(d, f, wrapped, 8, 6)
        txt = '\n'.join(wrapped)
        probes = [dict(id='passphrase', q="What passphrase is stated in the text?", a=pw),
                  dict(id='n_lines', q="How many lines of text are in the image?", a=str(len(wrapped))),
                  dict(id='verbatim_first', q="Transcribe the first two lines exactly.", a='\n'.join(wrapped[:2])),
                  dict(id='verbatim_last', q="Transcribe the last line exactly.", a=wrapped[-1]),
                  dict(id='comprehension', q="Summarise what the passage describes and list every "
                       "'the tally stood at N units against a quota of M' pair you can read.",
                       a='; '.join(x.strip() for x in facts if x.strip() in ' '.join(wrapped)) or
                         '(facts truncated - see groundtruth)')]
        emit(f"C_prose_proportional/C{i:02d}_dejavu_sans_{s:02d}px.png", im, txt,
             dict(series='C_prose_proportional', style='wrapped prose, proportional',
                  font='DejaVu Sans', size_px=s, antialiased=True, rows=len(wrapped)), probes)

# ============================================================ D: eye charts
def series_D():
    print("\n[D] eye charts - one image, every size, find the legibility floor")
    SENT = "Sphinx of black quartz judge my vow {n} times: code {c}"
    # D1 bitmap
    W, H = 896, 896
    im, d = canvas(W, H)
    rng = random.Random('D1'); y = 8; key = []
    for name in BITMAPS:
        f = bitmap(name)
        c = code(rng); n = rng.randint(10, 99)
        s = f"[{name}] " + SENT.format(n=n, c=c)
        d.text((8, y), s, font=f.font, fill=(0, 0, 0))
        key.append(s); y += f.lh + 6
    txt = '\n'.join(key)
    emit("D_eyecharts/D1_bitmap_eyechart.png", im, txt,
         dict(series='D_eyecharts', style='one line per bitmap size, ascending',
              font='X11 misc 4x6..12x24', antialiased=False, rows=len(key)),
         [dict(id='transcribe_all', q="Transcribe every line, smallest first. Say which lines you cannot read.", a=txt),
          dict(id='smallest_legible', q="What is the code on the very first (smallest) line?", a=key[0].split('code ')[1])])
    # D2 truetype
    im, d = canvas(W, H)
    rng = random.Random('D2'); y = 8; key = []
    for s in TTF_SIZES:
        f = ttf(DEJAVU_MONO, s)
        c = code(rng); n = rng.randint(10, 99)
        line = f"[{s}px] " + SENT.format(n=n, c=c)
        d.text((8, y), line, font=f.font, fill=(0, 0, 0))
        key.append(line); y += f.lh + 8
    txt = '\n'.join(key)
    emit("D_eyecharts/D2_truetype_eyechart.png", im, txt,
         dict(series='D_eyecharts', style='one line per px size, ascending',
              font='DejaVu Sans Mono 5..32px', antialiased=True, rows=len(key)),
         [dict(id='transcribe_all', q="Transcribe every line, smallest first. Say which lines you cannot read.", a=txt),
          dict(id='smallest_legible', q="What is the code on the very first (smallest) line?", a=key[0].split('code ')[1])])
    # D3 shrinking paragraph: same text repeated at decreasing size
    im, d = canvas(W, H); rng = random.Random('D3'); y = 6; key = []
    for name in reversed(BITMAPS):
        f = bitmap(name); cols, _ = fits(f, W, H)
        c = code(rng)
        s = (f"{name} {c} " + record(rng, cols - 12))[:cols]
        d.text((6, y), s, font=f.font, fill=(0, 0, 0))
        key.append(s); y += f.lh + 4
        if y > H - 30: break
    txt = '\n'.join(key)
    emit("D_eyecharts/D3_descending_fullwidth.png", im, txt,
         dict(series='D_eyecharts', style='full-width line per size, descending',
              font='X11 misc 12x24..4x6', antialiased=False, rows=len(key)),
         [dict(id='transcribe_all', q="Transcribe every line top to bottom.", a=txt),
          dict(id='last_line', q="Transcribe the final (smallest) line.", a=key[-1])])

if __name__ == '__main__':
    pass

# ============================================================ E: degradations
def _noise(im, sigma):
    try:
        import numpy as np
        a = np.asarray(im).astype('int16')
        rs = np.random.RandomState(7)
        a = a + rs.normal(0, sigma, a.shape).astype('int16')
        return Image.fromarray(a.clip(0, 255).astype('uint8'))
    except ImportError:
        return im

def series_E(W=896, H=896):
    print("\n[E] degradation stress tests (fixed 6x10 bitmap + 10px DejaVu baseline)")
    def page(fontobj, seed, pad=4):
        cols, rows = fits(fontobj, W, H, pad=pad)
        rng = random.Random(seed)
        lines, meta = numbered_lines(rng, cols, rows)
        return lines, meta, rows
    base_specs = [('6x10', bitmap('6x10'), 'E_bitmap6x10'), ('10px', ttf(DEJAVU_MONO, 10), 'E_mono10px')]
    for tag, f, seed in base_specs:
        lines, meta, rows = page(f, seed)
        probes = line_probes(lines, meta, rows)
        txt = '\n'.join(lines)
        def render(fg=(0,0,0), bg=(255,255,255)):
            im, d = canvas(W, H, bg); draw_lines(d, f, lines, 4, 4, fill=fg); return im
        variants = {}
        variants['00_baseline']      = render()
        variants['01_gray60']        = render(fg=(102,102,102))
        variants['02_gray30_faint']  = render(fg=(178,178,178))
        variants['03_inverted']      = render(fg=(235,235,235), bg=(12,12,12))
        variants['04_blue_on_cream'] = render(fg=(30,60,150), bg=(250,244,224))
        variants['05_red_on_green']  = render(fg=(200,40,40), bg=(40,150,60))
        variants['06_blur']          = render().filter(ImageFilter.GaussianBlur(0.6))
        variants['07_noise']         = _noise(render(), 14)
        variants['08_rotate1_5deg']  = render().rotate(1.5, resample=Image.BICUBIC, fillcolor=(255,255,255))
        variants['09_downup_50pct']  = render().resize((W//2, H//2), Image.LANCZOS).resize((W, H), Image.LANCZOS)
        variants['10_lowres_75pct']  = render().resize((int(W*.75), int(H*.75)), Image.LANCZOS)
        for k, im in variants.items():
            ext = 'png'
            emit(f"E_degradations/{tag}/E_{tag}_{k}.{ext}", im, txt,
                 dict(series='E_degradations', style=f'{k} on {tag} numbered lines',
                      font=f.label, degradation=k, antialiased=f.aa, rows=rows), probes)
        for q in (30, 60):
            im = render()
            p = os.path.join(OUT, f"E_degradations/{tag}/E_{tag}_11_jpeg{q}.jpg")
            os.makedirs(os.path.dirname(p), exist_ok=True); im.save(p, quality=q)
            emit(f"E_degradations/{tag}/E_{tag}_11_jpeg{q}.jpg", Image.open(p), txt,
                 dict(series='E_degradations', style=f'JPEG q={q}', font=f.label,
                      degradation=f'jpeg{q}', antialiased=f.aa, rows=rows), probes)

def series_E2():
    """Oversized page: exceeds Claude's ~1.15MP budget, so the API downscales it first."""
    print("\n[E2] oversized pages - demonstrates API-side downscaling destroying small text")
    for W, H, name in [(1568, 1568, '1568x1568_2.5MP'), (2240, 2240, '2240x2240_5MP')]:
        f = bitmap('6x10')
        cols, rows = fits(f, W, H)
        rng = random.Random('E2' + name)
        lines, meta = numbered_lines(rng, cols, rows)
        im, d = canvas(W, H); draw_lines(d, f, lines, 4, 4)
        emit(f"E_degradations/oversized/E2_{name}.png", im, '\n'.join(lines),
             dict(series='E_degradations', style='oversized - provider will downscale',
                  font=f.label, degradation='provider_downscale', antialiased=False, rows=rows),
             line_probes(lines, meta, rows))

# ============================================================ F: real-world layouts
SKU_W = ["hex bolt M8","copper lug 16mm","gasket ring 40","fuse 6.3A slow","relay 24V DPDT",
         "cable gland M20","bearing 6203ZZ","valve seat brass","o-ring 22x2","terminal block 12P",
         "sensor NTC 10k","contactor 32A","filter cartridge","pump seal kit","limit switch"]

def _table(f, W, H, rng, nrows, title=None, grid=False):
    cols = [('ROW',4),('SKU',9),('DESCRIPTION',20),('QTY',6),('UNIT',9),('TOTAL',10)]
    rows, tot_qty, tot_val = [], 0, 0
    for i in range(nrows):
        q = rng.randint(1, 400); u = rng.randint(50, 9999) / 100
        v = round(q * u, 2); tot_qty += q; tot_val += v
        rows.append([f"{i+1:03d}", code(rng, 7), rng.choice(SKU_W)[:20], str(q), f"{u:,.2f}", f"{v:,.2f}"])
    tot_val = round(tot_val, 2)
    hdr = ''.join(n.ljust(w) for n, w in cols)
    body = [''.join(c.ljust(w) for c, (n, w) in zip(r, cols)) for r in rows]
    foot = ''.join(x.ljust(w) for x, (n, w) in zip(['','','TOTAL', str(tot_qty), '', f"{tot_val:,.2f}"], cols))
    return hdr, body, foot, rows, tot_qty, tot_val

def series_F():
    print("\n[F] real-world layouts")
    # ---- F1/F2 data table
    for tag, f, W, H in [('tiny', bitmap('5x8'), 896, 896), ('normal', bitmap('9x18'), 896, 896)]:
        rng = random.Random('F_table_' + tag)
        cols, rws = fits(f, W, H, pad=6)
        n = rws - 4
        hdr, body, foot, raw, tq, tv = _table(f, W, H, rng, n)
        im, d = canvas(W, H)
        y = draw_lines(d, f, [hdr], 6, 6)
        d.line([(6, y + 1), (W - 6, y + 1)], fill=(0, 0, 0))
        y = draw_lines(d, f, body, 6, y + 3)
        d.line([(6, y + 1), (W - 6, y + 1)], fill=(0, 0, 0))
        draw_lines(d, f, [foot], 6, y + 3)
        txt = '\n'.join([hdr] + body + [foot])
        pick = [n // 5, n // 2, n - 1]
        probes = [dict(id='n_rows', q="How many data rows does the table have?", a=str(n)),
                  dict(id='total_qty', q="What is the TOTAL of the QTY column?", a=str(tq)),
                  dict(id='total_val', q="What is the TOTAL of the TOTAL column?", a=f"{tv:,.2f}"),
                  dict(id='sum_check', q="Add the QTY column yourself and say whether the printed TOTAL is correct.",
                       a=f"correct, {tq}")]
        for i in pick:
            probes.append(dict(id=f'row_{i+1:03d}', q=f"Give every cell of row {i+1:03d}.", a=' | '.join(raw[i])))
        emit(f"F_layouts/F1_table_{tag}_{f.label}.png", im, txt,
             dict(series='F_layouts', style='data table with column totals', font=f.label,
                  antialiased=f.aa, rows=n), probes)
    # ---- F3/F4 source code listing
    CODE = '''import hashlib, json, os
from dataclasses import dataclass, field

@dataclass
class LedgerEntry:
    seq: int
    sku: str
    qty: int
    unit_cents: int
    tags: list = field(default_factory=list)

    @property
    def total_cents(self) -> int:
        return self.qty * self.unit_cents

    def digest(self, salt: bytes = b"") -> str:
        blob = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.blake2b(blob + salt, digest_size=16).hexdigest()

class Ledger:
    """Append-only ledger with a rolling checksum."""

    def __init__(self, path, *, strict=True):
        self.path = path
        self.strict = strict
        self._entries = []
        self._checksum = "0" * 32

    def append(self, entry: LedgerEntry) -> str:
        if self.strict and entry.qty <= 0:
            raise ValueError(f"non-positive qty on seq {entry.seq}")
        if any(e.seq == entry.seq for e in self._entries):
            raise KeyError(f"duplicate seq {entry.seq}")
        self._entries.append(entry)
        self._checksum = entry.digest(self._checksum.encode())
        return self._checksum

    def total_cents(self) -> int:
        return sum(e.total_cents for e in self._entries)

    def by_tag(self, tag):
        return [e for e in self._entries if tag in e.tags]

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            for e in self._entries:
                fh.write(json.dumps(e.__dict__) + "\\n")
        os.replace(tmp, self.path)
        return len(self._entries)
'''.strip('\n').split('\n')
    for tag, f in [('tiny', bitmap('6x10')), ('normal', ttf(DEJAVU_MONO, 16))]:
        W, H = 896, 896
        numbered = [f"{i+1:3d} | {l}" for i, l in enumerate(CODE)]
        im, d = canvas(W, H, (253, 253, 250))
        draw_lines(d, f, numbered, 8, 8)
        txt = '\n'.join(numbered)
        probes = [dict(id='line_22', q="Transcribe line 22 exactly (including indentation).", a=numbered[21]),
                  dict(id='methods', q="List every method defined on class Ledger, in order.",
                       a="__init__, append, total_cents, by_tag, flush"),
                  dict(id='count_return', q="How many lines contain the keyword 'return'?",
                       a=str(sum(1 for l in CODE if 'return' in l))),
                  dict(id='exception', q="What exception is raised for a duplicate seq, and on what line number?",
                       a=f"KeyError, line {[i+1 for i,l in enumerate(CODE) if 'KeyError' in l][0]}"),
                  dict(id='digest_size', q="What digest_size is passed to blake2b?", a="16")]
        emit(f"F_layouts/F2_code_{tag}_{f.label}.png", im, txt,
             dict(series='F_layouts', style='source code listing with line numbers', font=f.label,
                  antialiased=f.aa, rows=len(numbered)), probes)

# ---- F3 newspaper (3-column proportional)
NEWS_H = "YARD AUDIT FINDS ELEVEN HOUR DELAY AT NORTH GATE"
def series_F3():
    print("\n[F3] three-column newspaper page")
    for tag, s, W, H in [('tiny', 7, 896, 896), ('normal', 13, 896, 896)]:
        f  = ttf(DEJAVU_SANS, s, mono=False)
        fh = ttf(DEJAVU_BOLD, max(18, s * 3), mono=False)
        fs = ttf(DEJAVU_BOLD, max(9, s + 2), mono=False)
        rng = random.Random('F3' + tag)
        im, d = canvas(W, H, (252, 251, 246))
        d.text((16, 10), NEWS_H, font=fh.font, fill=(0, 0, 0))
        y0 = 14 + fh.lh + 6
        d.line([(16, y0 - 6), (W - 16, y0 - 6)], fill=(0, 0, 0), width=2)
        colw = (W - 32 - 2 * 18) // 3
        all_lines, facts_all = [NEWS_H], []
        for c in range(3):
            x = 16 + c * (colw + 18)
            body, facts = prose(rng, 4000)
            facts_all += facts
            sub = ["NORTH GATE", "WEIGHBRIDGE", "THE LEDGER"][c]
            d.text((x, y0), sub, font=fs.font, fill=(0, 0, 0))
            lines = wrap_to(f, body, colw)[:int((H - y0 - fs.lh - 14) // f.lh)]
            draw_lines(d, f, lines, x, y0 + fs.lh + 3)
            if c < 2: d.line([(x + colw + 9, y0), (x + colw + 9, H - 14)], fill=(170, 170, 170))
            all_lines += [f"--- COLUMN {c+1}: {sub} ---"] + lines
        txt = '\n'.join(all_lines)
        vis = ' '.join(all_lines)
        probes = [dict(id='headline', q="What is the headline?", a=NEWS_H),
                  dict(id='subheads', q="List the three column sub-headings in order.", a="NORTH GATE; WEIGHBRIDGE; THE LEDGER"),
                  dict(id='col3_first', q="Transcribe the first line of the third column.",
                       a=all_lines[all_lines.index('--- COLUMN 3: THE LEDGER ---') + 1]),
                  dict(id='facts', q="List every 'tally stood at N units against a quota of M' pair, per column.",
                       a='; '.join(x.strip() for x in facts_all if x.split('.')[0].strip()[:40] in vis)),
                  dict(id='comprehension', q="What does the article describe, and what caused the delay?",
                       a="a yard/manifest inspection audit at the north gate; a mislabelled pallet holds a convoy for eleven hours")]
        emit(f"F_layouts/F3_newspaper_{tag}_{s}px.png", im, txt,
             dict(series='F_layouts', style='3-column newspaper, proportional', font=f'DejaVu Sans {s}px + bold heads',
                  antialiased=True, rows=len(all_lines)), probes)

# ---- F4 receipt (narrow tall)
def series_F4():
    print("\n[F4] narrow receipts")
    for tag, f in [('tiny', bitmap('4x6')), ('small', bitmap('6x10')), ('normal', bitmap('9x18'))]:
        W, H = 448, 1344
        rng = random.Random('F4' + tag)
        cols, rows = fits(f, W, H, pad=6)
        n = min(rows - 16, 90)
        items, sub = [], 0.0
        for i in range(n):
            q = rng.randint(1, 9); u = rng.randint(45, 4999) / 100
            v = round(q * u, 2); sub += v
            name = rng.choice(SKU_W)
            items.append((f"{q}x {name}"[:cols - 10], f"{v:>8.2f}", v, name, q))
        sub = round(sub, 2); tax = round(sub * 0.0825, 2); tot = round(sub + tax, 2)
        L = ["  BRIGHTON YARD SUPPLY CO", "  1174 DOCK ROAD, UNIT 6", f"  TERMINAL 04  REG 12  OP {code(rng,4)}",
             f"  {rng.randint(1,28):02d}/08/2026  14:{rng.randint(10,59)}:{rng.randint(10,59)}", "-" * cols]
        for nm, pr, *_ in items: L.append(nm.ljust(cols - 9) + pr)
        L += ["-" * cols, "SUBTOTAL".ljust(cols - 9) + f"{sub:>8.2f}",
              "TAX 8.25%".ljust(cols - 9) + f"{tax:>8.2f}", "TOTAL".ljust(cols - 9) + f"{tot:>8.2f}",
              "CARD **** 4417  APPROVED", f"AUTH {code(rng,6)}  REF {rng.randint(10**7,10**8)}", "-" * cols,
              "RETURNS ACCEPTED WITHIN 30 DAYS WITH", "RECEIPT. NO REFUND ON CUT CABLE OR",
              "SPECIAL ORDER ITEMS. SEE REVERSE FOR", "FULL TERMS.  QUERIES: 0800 114 9930"]
        im, d = canvas(W, H, (255, 255, 252)); draw_lines(d, f, L, 6, 6)
        k = n // 2
        probes = [dict(id='total', q="What is the TOTAL?", a=f"{tot:.2f}"),
                  dict(id='tax', q="What tax rate and tax amount are shown?", a=f"8.25%, {tax:.2f}"),
                  dict(id='n_items', q="How many line items are on the receipt?", a=str(n)),
                  dict(id='item_k', q=f"What is line item {k+1} (from the top) and its price?",
                       a=f"{items[k][0]} = {items[k][1].strip()}"),
                  dict(id='arith', q="Add the line items yourself: does SUBTOTAL match?", a=f"yes, {sub:.2f}"),
                  dict(id='fineprint', q="Transcribe the small-print returns policy.",
                       a="RETURNS ACCEPTED WITHIN 30 DAYS WITH RECEIPT. NO REFUND ON CUT CABLE OR SPECIAL ORDER ITEMS. SEE REVERSE FOR FULL TERMS. QUERIES: 0800 114 9930")]
        emit(f"F_layouts/F4_receipt_{tag}_{f.label}.png", im, '\n'.join(L),
             dict(series='F_layouts', style='narrow receipt, right-aligned prices', font=f.label,
                  antialiased=f.aa, rows=len(L)), probes)

# ---- F5 terminal log (dark, coloured levels)
def series_F5():
    print("\n[F5] dark terminal logs")
    SVC = ["ingest","router","ledger","auth","cache","sweeper","planner","uploader"]
    MSG = ["connection reset by peer","retry scheduled in {n}ms","flushed {n} records",
           "checksum mismatch on shard {n}","lease renewed for {n}s","queue depth {n}",
           "dropping stale frame {n}","reconnected to node-{n}","backpressure engaged at {n}%",
           "compaction finished in {n}ms","token refresh failed, code {n}"]
    for tag, f in [('tiny', bitmap('5x8')), ('small', bitmap('6x13')), ('normal', bitmap('9x18'))]:
        W, H = 1344, 672
        rng = random.Random('F5' + tag)
        cols, rows = fits(f, W, H, pad=6)
        im, d = canvas(W, H, (18, 18, 22))
        COL = {'INFO': (200, 205, 210), 'WARN': (222, 186, 60), 'ERROR': (232, 84, 74), 'DEBUG': (110, 118, 130)}
        L, errs = [], []
        for i in range(rows):
            lv = rng.choices(['INFO','DEBUG','WARN','ERROR'], [6,3,2,1])[0]
            s = (f"2026-08-{rng.randint(10,18):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:"
                 f"{rng.randint(0,59):02d}.{rng.randint(100,999)}Z [{lv:<5}] pid={rng.randint(1000,9999)} "
                 f"{rng.choice(SVC):<8} {rng.choice(MSG).format(n=rng.randint(1,9999))}")[:cols]
            d.text((6, 6 + i * f.lh), s, font=f.font, fill=COL[lv])
            L.append(s)
            if lv == 'ERROR': errs.append((i + 1, s))
        probes = [dict(id='n_errors', q="How many ERROR lines are there?", a=str(len(errs))),
                  dict(id='first_error', q="Transcribe the first ERROR line.", a=errs[0][1] if errs else 'none'),
                  dict(id='error_pids', q="List the pid of every ERROR line, in order.",
                       a=', '.join(e[1].split('pid=')[1].split()[0] for e in errs)),
                  dict(id='line_20', q="Transcribe line 20 (counting from the top).", a=L[19]),
                  dict(id='services', q="Which distinct service names appear?",
                       a=', '.join(sorted({l.split(']')[1].split()[1] for l in L})))]
        emit(f"F_layouts/F5_terminal_{tag}_{f.label}.png", im, '\n'.join(L),
             dict(series='F_layouts', style='dark terminal log, coloured levels', font=f.label,
                  antialiased=f.aa, rows=rows), probes)

# ---- F6 spreadsheet grid (2D cell addressing)
def series_F6():
    print("\n[F6] spreadsheet grids - 2D cell addressing")
    for tag, f in [('tiny', bitmap('5x8')), ('small', bitmap('6x10')), ('normal', bitmap('8x16'))]:
        W, H = 896, 896
        rng = random.Random('F6' + tag)
        cellw, rowh = 8 * f.cw, f.lh + 3
        ncol = min(16, (W - 40) // cellw); nrow = min(48, (H - 20) // rowh - 1)
        letters = [chr(65 + i) for i in range(ncol)]
        vals = [[rng.randint(-999, 9999) for _ in range(ncol)] for _ in range(nrow)]
        im, d = canvas(W, H)
        d.rectangle([0, 0, W, rowh + 4], fill=(232, 232, 236))
        d.rectangle([0, 0, 36, H], fill=(232, 232, 236))
        for c in range(ncol + 1): d.line([(36 + c * cellw, 0), (36 + c * cellw, H)], fill=(198, 198, 205))
        for r in range(nrow + 2): d.line([(0, 4 + r * rowh), (W, 4 + r * rowh)], fill=(198, 198, 205))
        for c, L in enumerate(letters):
            d.text((36 + c * cellw + cellw // 2 - f.cw // 2, 6), L, font=f.font, fill=(40, 40, 40))
        lines = ['     ' + ''.join(l.rjust(8) for l in letters)]
        for r in range(nrow):
            y = 8 + rowh + r * rowh
            d.text((6, y), f"{r+1:>4}", font=f.font, fill=(40, 40, 40))
            for c in range(ncol):
                s = f"{vals[r][c]:,}"
                d.text((36 + (c + 1) * cellw - 3 - f.cw * len(s), y), s, font=f.font, fill=(0, 0, 0))
            lines.append(f"{r+1:>4} " + ''.join(f"{v:,}".rjust(8) for v in vals[r]))
        tc, tr = min(5, ncol - 1), min(16, nrow - 1)
        colsum = sum(vals[r][2] for r in range(nrow)) if ncol > 2 else 0
        rowmax = max(vals[min(8, nrow - 1)])
        probes = [dict(id='dims', q="How many labelled columns and numbered rows does the sheet have?",
                       a=f"{ncol} columns (A-{letters[-1]}), {nrow} rows"),
                  dict(id='cell', q=f"What value is in cell {letters[tc]}{tr+1}?", a=f"{vals[tr][tc]:,}"),
                  dict(id='cell2', q=f"What value is in cell A1 and in cell {letters[-1]}{nrow}?",
                       a=f"A1={vals[0][0]:,}; {letters[-1]}{nrow}={vals[nrow-1][ncol-1]:,}"),
                  dict(id='colsum', q="What is the sum of column C?", a=f"{colsum:,}"),
                  dict(id='rowmax', q=f"What is the largest value in row {min(9, nrow)}?", a=f"{rowmax:,}"),
                  dict(id='negatives', q="How many negative numbers are in column B?",
                       a=str(sum(1 for r in range(nrow) if vals[r][1] < 0)))]
        emit(f"F_layouts/F6_spreadsheet_{tag}_{f.label}.png", im, '\n'.join(lines),
             dict(series='F_layouts', style='spreadsheet grid, row/col headers', font=f.label,
                  antialiased=f.aa, rows=nrow, cols=ncol), probes)

# ============================================================ G: one page, many sizes
def series_G():
    print("\n[G] multi-scale documents - title down to fine print in a single image")
    specs = [
      ('G1_contract', 896, 1120, [
        ('title',   ttf(DEJAVU_BOLD, 30, mono=False), 'SUPPLY AGREEMENT - SCHEDULE 4'),
        ('heading', ttf(DEJAVU_BOLD, 16, mono=False), 'PART A: DELIVERY OBLIGATIONS'),
        ('body',    ttf(DEJAVU_SANS, 11, mono=False), None),
        ('notes',   ttf(DEJAVU_SANS,  8, mono=False), None),
        ('fine',    bitmap('5x7'),                    None),
        ('micro',   bitmap('4x6'),                    None)]),
      ('G2_label', 896, 896, [
        ('title',   ttf(DEJAVU_BOLD, 24, mono=False), 'CONTENTS: 500 ML  BATCH 7741-K'),
        ('heading', ttf(DEJAVU_BOLD, 13, mono=False), 'DIRECTIONS FOR USE'),
        ('body',    ttf(DEJAVU_SANS, 10, mono=False), None),
        ('notes',   ttf(DEJAVU_SANS,  7, mono=False), None),
        ('fine',    bitmap('4x6'),                    None)]),
    ]
    for name, W, H, tiers in specs:
        rng = random.Random(name)
        im, d = canvas(W, H, (255, 255, 253))
        y, txt, probes = 12, [], []
        share = (H - 40) / sum(1.6 if t[0] in ('body','notes') else 1 for t in tiers)
        for tier, f, fixed in tiers:
            tcode = code(rng, 6)
            if fixed:
                head = f"{fixed}  [{tier.upper()} CODE {tcode}]"
                hl = wrap_to(f, head, W - 24)
                draw_lines(d, f, hl, 12, y); txt += hl
                y += len(hl) * f.lh + 8
            else:
                budget = int(share * (1.6 if tier in ('body', 'notes') else 1))
                nlines = max(1, budget // f.lh)
                body, _ = prose(rng, 600 + nlines * int((W - 24) / max(1, f.cw * 0.9)))
                body = f"[{tier.upper()} CODE {tcode}] " + body
                lines = wrap_to(f, body, W - 24)[:nlines]
                draw_lines(d, f, lines, 12, y); txt += lines
                y += nlines * f.lh + 10
            probes.append(dict(id=f'code_{tier}', q=f"What is the {tier.upper()} CODE?", a=tcode,
                               tier=tier, font=f.label))
            if y > H - 20: break
        probes.insert(0, dict(id='tiers', q="How many distinct text sizes appear, and what does each contain?",
                              a=f"{len(tiers)} tiers: " + ', '.join(t[0] for t in tiers)))
        probes.append(dict(id='smallest_verbatim', q="Transcribe the first line of the SMALLEST text.",
                           a=txt[-1] if txt else ''))
        emit(f"G_multiscale/{name}.png", im, '\n'.join(txt),
             dict(series='G_multiscale', style='single page, cascading text sizes',
                  font=' / '.join(t[1].label for t in tiers), antialiased=True, rows=len(txt)), probes)

# ============================================================ build everything
def main():
    for fn in (series_A, series_B, series_B2, series_C, series_D,
               series_E, series_E2, series_F, series_F3, series_F4, series_F5, series_F6, series_G):
        fn()
    # ---- answer key
    key = dict(
        generated_by='generate.py',
        patch_model=dict(claude_px_per_token=CLAUDE_PATCH, gpt_5_6_px_per_token=GPT_PATCH,
                         claude_downscale_above_pixels=CLAUDE_MAX_PIXELS,
                         note='canvas sizes are multiples of 224 = lcm(28,32) so both grids divide evenly'),
        images=ENTRIES)
    with open(os.path.join(ROOT, 'ANSWER_KEY.json'), 'w') as f:
        json.dump(key, f, indent=1)
    # ---- manifest
    cols = ['file','series','style','font','cell','size_px','cap_height_px','antialiased','degradation',
            'w','h','megapixels','rows','cols','chars','claude_tokens','gpt_tokens',
            'chars_per_claude_token','chars_per_gpt_token','text_tokens_equiv','claude_compression',
            'oversized_for_claude','groundtruth']
    with open(os.path.join(ROOT, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader()
        for e in ENTRIES.values(): w.writerow(e)
    print(f"\n{len(ENTRIES)} images -> {OUT}")
    print(f"answer key  -> {os.path.join(ROOT,'ANSWER_KEY.json')}")
    print(f"manifest    -> {os.path.join(ROOT,'manifest.csv')}")
    return ENTRIES

if __name__ == '__main__':
    main()
