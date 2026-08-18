#!/usr/bin/env python3
"""Shared helpers: fonts, token math, synthetic corpus."""
import os, re, gzip, shutil, math, random, textwrap
from PIL import Image, ImageDraw, ImageFont, PcfFontFile

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT     = os.path.join(ROOT, 'images')
GT      = os.path.join(ROOT, 'groundtruth')
FONTDIR = os.path.join(ROOT, '_pilfonts')
X11     = '/usr/share/fonts/X11/misc'
DEJAVU_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
DEJAVU_SANS = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
DEJAVU_COND = '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf'
DEJAVU_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
MONO_BOLD   = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'

CLAUDE_PATCH = 28   # Claude: 28x28 px per image token
GPT_PATCH    = 32   # GPT-5.6: 32x32 px per image token
CLAUDE_MAX_PIXELS = 1_150_000   # above this Claude downscales, destroying tiny text

for d in (OUT, GT, FONTDIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- fonts
class F:
    """Uniform wrapper over bitmap + truetype fonts."""
    def __init__(self, font, cw, lh, label, mono=True, aa=True):
        self.font, self.cw, self.lh, self.label, self.mono, self.aa = font, cw, lh, label, mono, aa
    def width(self, s):
        if self.mono: return self.cw * len(s)
        return self.font.getlength(s)

class _FixedPcf(PcfFontFile.PcfFontFile):
    """Pillow indexes the PCF encoding table by absolute codepoint, ignoring
    first_col.  X11's 8x16 and 12x24 declare first_col=1, so every glyph comes
    out shifted by one.  This re-implements the lookup relative to first_col."""
    def _load_encoding(self):
        fp, fmt, i16, i32 = self._getformat(PcfFontFile.PCF_BDF_ENCODINGS)
        first_col, last_col = i16(fp.read(2)), i16(fp.read(2))
        first_row, last_row = i16(fp.read(2)), i16(fp.read(2))
        i16(fp.read(2))                       # default char
        n = (last_col - first_col + 1) * (last_row - first_row + 1)
        offsets = [i16(fp.read(2)) for _ in range(n)]
        enc = [None] * 256
        for c in range(first_col, min(256, last_col + 1)):
            i = c - first_col                 # <- the fix
            if i < len(offsets) and offsets[i] != 0xFFFF:
                enc[c] = offsets[i]
        return enc

_pcf_cache = {}
def bitmap(name):
    """X11 misc bitmap font, e.g. '4x6'. Pixel-exact, no anti-aliasing."""
    if name in _pcf_cache: return _pcf_cache[name]
    dst = os.path.join(FONTDIR, name + '.pil')
    if not os.path.exists(dst):
        tmp = os.path.join(FONTDIR, name + '.pcf')
        with gzip.open(os.path.join(X11, name + '.pcf.gz')) as f, open(tmp, 'wb') as o:
            shutil.copyfileobj(f, o)
        with open(tmp, 'rb') as fp:
            _FixedPcf(fp).save(os.path.join(FONTDIR, name))
        os.remove(tmp)
    m = re.search(r'(\d+)x(\d+)', name)
    cw, lh = int(m.group(1)), int(m.group(2))
    f = F(ImageFont.load(dst), cw, lh, name, mono=True, aa=False)
    _pcf_cache[name] = f
    return f

def ttf(path, size, mono=True, tight=1.0, label=None):
    fo = ImageFont.truetype(path, size)
    asc, desc = fo.getmetrics()
    lh = max(size + 1, int(round((asc + desc) * tight)))
    cw = fo.getlength('0')
    return F(fo, cw, lh, label or f'{os.path.basename(path)[:-4]}@{size}px', mono=mono)

# ---------------------------------------------------------------- token math
def tokens(w, h):
    c = math.ceil(w / CLAUDE_PATCH) * math.ceil(h / CLAUDE_PATCH)
    g = math.ceil(w / GPT_PATCH)    * math.ceil(h / GPT_PATCH)
    return c, g

def stats(w, h, text):
    chars = len(text.replace('\n', ''))
    ct, gt = tokens(w, h)
    return dict(w=w, h=h, chars=chars, lines=text.count('\n') + 1,
                megapixels=round(w * h / 1e6, 3),
                claude_tokens=ct, gpt_tokens=gt,
                chars_per_claude_token=round(chars / ct, 2),
                chars_per_gpt_token=round(chars / gt, 2),
                text_tokens_equiv=round(chars / 4),
                claude_compression=round((chars / 4) / ct, 2),
                oversized_for_claude=(w * h > CLAUDE_MAX_PIXELS))

# ---------------------------------------------------------------- corpus
SUBJ = ["Depot Kilo","Sector 14","Unit Bravo","The north yard","Terminal 7","Crew Delta",
        "Warehouse 3","The relay station","Convoy Echo","Platform 22","Substation Foxtrot",
        "The east pier","Team Juliet","Node 41","The cold store","Berth 19","Line Charlie"]
VERB = ["shipped","logged","rejected","rerouted","inspected","invoiced","recovered","quarantined",
        "audited","dispatched","weighed","relabelled","transferred","scanned","impounded"]
GOODS= ["crates of tin","pallets of resin","drums of glycol","spools of copper","bales of flax",
        "canisters of argon","cases of ballast","sacks of silica","reels of fibre","tubs of pitch",
        "kegs of solvent","bins of scrap","rolls of felt","trays of quartz","boxes of ferrite"]
TAIL = ["under seal {S}","against order {O}","via gate {G}","with manifest {M}","at bay {B}",
        "pending review {R}","on route {T}","for client {C}","against quota {Q}","by lift {L}"]
CLAUSE = ["temperature held at {t}C","humidity {hh} percent","no damage reported","two seals broken",
          "checksum {ck} verified","weight variance {wv} kg","held {hd} minutes for customs",
          "driver badge {db}","tare recorded as {tr} kg","photograph {ph} attached",
          "signature captured on pad {sp}","gate camera {gc} clear","manifest reprinted twice",
          "barcode {bc} unreadable on first pass","escort vehicle {ev} assigned"]
ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # no I/O/0/1 -> unambiguous grading

def code(rng, n=5):
    return ''.join(rng.choice(ALNUM) for _ in range(n))

def record(rng, width, rare=None):
    """One factual line, grown with clauses until it fills `width` chars."""
    parts = [f"{rng.choice(SUBJ)} {rng.choice(VERB)} {rng.randint(100,9999)} {rng.choice(GOODS)} "
             + rng.choice(TAIL).format(S=code(rng,4), O=rng.randint(1000,9999), G=rng.randint(1,48),
                                       M=code(rng,4), B=rng.randint(1,99), R=code(rng,3),
                                       T=rng.randint(1,9), C=code(rng,4), Q=rng.randint(10,99),
                                       L=rng.randint(1,12))]
    if rare: parts.append(rare)
    while sum(len(p) + 2 for p in parts) < width + 24:
        parts.append(rng.choice(CLAUSE).format(t=rng.randint(-8,34), hh=rng.randint(10,90),
                     ck=code(rng,4), wv=rng.randint(1,90), hd=rng.randint(5,240), db=code(rng,4),
                     tr=rng.randint(50,900), ph=code(rng,3), sp=rng.randint(1,9), gc=rng.randint(1,30),
                     bc=code(rng,6), ev=code(rng,3)))
    s = '; '.join(parts)
    if len(s) > width:
        s = s[:width]
        if ' ' in s[-14:]: s = s[:s.rstrip().rfind(' ')]
    return s

def numbered_lines(rng, cols, rows, needle_word="ZEPHYR"):
    """Dense addressable page: '0042 QK7M9 <text>'. Returns (lines, probes-dict)."""
    lines, meta = [], {}
    body = max(8, cols - 11)
    rare_rows = set(rng.sample(range(rows), min(rows, max(2, rows // 25)))) if rows > 3 else set()
    needle_row = rng.randrange(rows // 4, max(rows // 4 + 1, rows - 2)) if rows > 6 else rows - 1
    for i in range(rows):
        n = f"{i+1:04d}"
        c = code(rng)
        if i == needle_row:
            pw = f"{rng.choice(['VIOLET','AMBER','COBALT','SORREL','INDIGO'])}-" \
                 f"{rng.choice(['BADGER','MARLIN','FALCON','OTTER','LYNX'])}-{rng.randint(1000,9999)}"
            txt = (f"!! PASSPHRASE {pw} !! " + record(rng, max(4, body - 26)))[:body]
            meta['passphrase'] = pw
        else:
            txt = record(rng, body, rare=(needle_word if i in rare_rows else None))
        lines.append(f"{n} {c} {txt}"[:cols])
    meta['needle_word'] = needle_word
    meta['needle_count'] = sum(1 for l in lines if needle_word in l)
    meta['needle_lines'] = [i + 1 for i, l in enumerate(lines) if needle_word in l]
    meta['passphrase_line'] = needle_row + 1
    return lines, meta

PROSE_SEED = (
 "The inspection began at the north gate, where two auditors compared the printed manifest against "
 "the physical count. Discrepancies were rare but consequential: a single mislabelled pallet could "
 "hold up an entire convoy for eleven hours. The senior auditor, working from a clipboard, recorded "
 "each variance in a ruled notebook and initialled the margin. Later that afternoon the readings "
 "were transcribed into the ledger, cross-checked against the weighbridge tape, and filed. "
 "Nothing in the procedure was novel, yet the discipline of it kept the yard solvent through a "
 "decade of thin margins, three regulatory reviews, and one memorable flood. ")

def prose(rng, nchars, facts=True):
    """Readable prose with injected verifiable facts, ~nchars long."""
    out, injected = [], []
    while sum(len(x) for x in out) < nchars:
        out.append(PROSE_SEED)
        if facts:
            f = (f"On {rng.randint(1,28)} {rng.choice(['March','June','August','November'])} the tally "
                 f"stood at {rng.randint(1000,9999)} units against a quota of {rng.randint(1000,9999)}, "
                 f"and inspector {code(rng,3)}-{rng.randint(10,99)} signed off at "
                 f"{rng.randint(6,19)}:{rng.randint(10,59)}. ")
            injected.append(f)
            out.append(f)
    return ''.join(out)[:nchars], injected

def wrap_to(font, text, maxw):
    """Greedy wrap of `text` to pixel width maxw."""
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if font.width(t) <= maxw: cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

# ---------------------------------------------------------------- drawing
def canvas(w, h, bg=(255, 255, 255)):
    im = Image.new('RGB', (w, h), bg)
    return im, ImageDraw.Draw(im)

def draw_lines(d, font, lines, x, y, fill=(0, 0, 0), lh=None, aa=True):
    if not aa: d.fontmode = "1"
    lh = lh or font.lh
    for i, ln in enumerate(lines):
        d.text((x, y + i * lh), ln, font=font.font, fill=fill)
    if not aa: d.fontmode = "L"
    return y + len(lines) * lh

def fits(font, w, h, pad=4):
    cols = int((w - 2 * pad) // font.cw)
    rows = int((h - 2 * pad) // font.lh)
    return cols, rows
