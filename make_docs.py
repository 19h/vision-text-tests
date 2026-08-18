#!/usr/bin/env python3
"""Build README.md + index.html from ANSWER_KEY.json / manifest.csv."""
import json, os, html, collections
ROOT = os.path.dirname(os.path.abspath(__file__))
K = json.load(open(os.path.join(ROOT, 'ANSWER_KEY.json')))
E = K['images']

def row(e, name):
    return (f"| {name} | {e.get('cap_height_px') or e.get('size_px') or '-'} | {e['chars']:,} | "
            f"{e['claude_tokens']:,} | {e['chars_per_claude_token']:.1f} | "
            f"{e['chars_per_gpt_token']:.1f} | {e['claude_compression']:.1f}x |")

A = [(k, v) for k, v in E.items() if v['series'] == 'A_bitmap_ladder']
B = [(k, v) for k, v in E.items() if v['series'] == 'B_mono_ttf_ladder' and 'noAA' not in k]
C = [(k, v) for k, v in E.items() if v['series'] == 'C_prose_proportional']
counts = collections.Counter(v['series'] for v in E.values())

HDR = ("| font | glyph px | chars in image | Claude img-tokens | chars / Claude token | "
       "chars / GPT token | vs. plain-text tokens |\n|---|---|---|---|---|---|---|")

readme = f"""# Dense-text vision benchmark

{len(E)} images for probing how small text can get before a vision LLM stops reading it,
plus a machine-checkable answer key for every one.

> **Status: exploratory, not confirmatory.** The prompt, tool policy, grader, payload
> families, line lengths and candidate font set were all changed *during* the run in
> response to what the results showed, and the 0.95 reporting threshold was chosen after
> seeing scores. That is legitimate discovery work, but it means these numbers carry
> adaptive-selection and winner's-curse risk. Regrading stored answers fixes grader bugs;
> it cannot fix a prompt or payload confound in the original call. `confirm.py` is the
> frozen re-run that addresses this - see **Confirmatory pass**.

Built around the image-tokeniser geometry you gave:

| model | pixels per image token | tokens for a 896x896 page |
|---|---|---|
| Claude | 28 x 28 | 1,024 |
| GPT-5.6 | 32 x 32 | 784 |

**Canvas dimensions here are multiples of 224 = lcm(28, 32)**, so one image divides evenly
into both grids and per-token density is comparable across the two models. Note that 224 is
only required if you need *both* alignments: for Claude alone, zero waste needs

    W = 0 mod lcm(28, cw)      H = 0 mod lcm(28, cell_height)

which is often much finer. For a 5x10 cell that is lcm(28,5) = lcm(28,10) = 140, so a
1540x1540 page is exactly aligned and reaches 784/50 = 15.68 chars/token, slightly denser
than the 1568 page. `pack.py` and `pack_text.py` search both alignments; only this README's
earlier wording implied 224 was mandatory. It also means **maximum chars/token and maximum
chars/image are different objectives** and can select different canvases.

### Measured, not assumed

Billed input tokens for the same prompt across four image sizes (`claude -p --output-format json`):

| image | pixels | billed input tokens | marginal px / token |
|---|---|---|---|
| 224 x 224 | 50,176 | 227 | - |
| 448 x 448 | 200,704 | 392 | 912 |
| 896 x 896 | 802,816 | 1,174 | **770** |
| 1344 x 1344 | 1,806,336 | 2,450 | **786** |

The marginal rate converges on 784 = 28 x 28, so the premise holds and a 896x896 page really
does cost ~1,024 image tokens.

Downscaling is *later* than the commonly-quoted ~1.15 MP: a 1568x1568 page (2.46 MP) billed
3,159 image tokens, the full `w*h/784` with no resampling. A 2240x2240 page (5.0 MP) billed
~4,833 instead of a nominal 6,400, implying an effective long edge near 1,950px. Pages here
stay at or below ~1 MP anyway, so the ladder measures the model rather than a resampler -
`oversized/` exists to show the difference.

## Start here

1. `images/D_eyecharts/D1_bitmap_eyechart.png` - every size in one image, ascending. One
   request tells you roughly where the floor is.
2. `images/A_bitmap_ladder/` - walk down the ladder until transcription breaks.
3. `images/E_degradations/` - take the size that just worked and degrade it.
4. `images/F_layouts/` + `images/G_multiscale/` - does the answer hold up on real documents?

## The ladders

### A - pixel-perfect bitmap (X11 misc), no anti-aliasing, 896x896

{HDR}
{chr(10).join(row(v, v['cell']) for k, v in A)}

### B - anti-aliased TrueType mono (DejaVu Sans Mono), 896x896

{HDR}
{chr(10).join(row(v, str(v['size_px']) + 'px') for k, v in B)}

### C - proportional prose (DejaVu Sans), 896x896

{HDR}
{chr(10).join(row(v, str(v['size_px']) + 'px') for k, v in C)}

The last column is the interesting one: at 4x6 a page holds ~8,000 plain-text tokens' worth
of characters in 1,024 image tokens. If a model can actually read it, images are an 8x cheaper
channel than text - and the ladder tells you exactly where that stops being true.

## Series

| dir | n | what it isolates |
|---|---|---|
| `A_bitmap_ladder` | {counts['A_bitmap_ladder']} | glyph size, with anti-aliasing removed as a variable. 4x6 up to 12x24. |
| `B_mono_ttf_ladder` | {counts['B_mono_ttf_ladder']} | the same ladder in anti-aliased TrueType, 5px..32px, plus 1-bit `noAA` twins at 6/8/10/12/16px. |
| `C_prose_proportional` | {counts['C_prose_proportional']} | proportional prose - real reading rather than code-lookup, and higher chars/px than mono. |
| `D_eyecharts` | {counts['D_eyecharts']} | every size in a single image; finds the floor in one request. |
| `E_degradations` | {counts['E_degradations']} | contrast, inversion, colour, blur, noise, 1.5deg rotation, resampling, JPEG q30/q60 - and two oversized pages the provider will downscale for you. |
| `F_layouts` | {counts['F_layouts']} | tables with column totals, code listings, 3-column newsprint, narrow receipts, dark terminal logs, spreadsheet grids with A1-style addressing. |
| `G_multiscale` | {counts['G_multiscale']} | one page cascading from a 30px title to 4x6 micro-print, each tier carrying its own code - shows where reading fails *within* one image. |

## Methodology: what these numbers are not

**The corpus is generated from a closed template, and that inflates legibility scores.**
Every line in series A-J comes from `record()`: 17 subjects x 15 verbs x 15 goods x 10
tails x 15 clauses. Once a model resolves a few lines it can infer the generator and
reconstruct the rest by pattern-matching - "Depot Kilo quarantined 5058 cases of ballast
on route 7" is recoverable from partial glyph evidence because the vocabulary is tiny and
the grammar is fixed. So a high verbatim score at a marginal size means *reading with
strong priors*, not glyph resolution.

This is why the random 5-character codes exist and why they score lower than verbatim
similarity everywhere: `ALNUM` codes are unguessable, so they cannot be reconstructed
from context. Series K re-renders the same geometry with an unguessable payload (random
4-character groups); **the gap between a J page and its K twin is the redundancy credit**,
and only the K number is a true glyph-resolution floor.

The same contamination applies to the person or model running the benchmark. While
analysing these results I read a crop of a marginal page and judged it legible - but I
had written the generator earlier in the same session, so its vocabulary was already in
my context. That is recall with a visual hint, not a legibility measurement. Any rater
who has seen the ground truth, the generator, or enough sibling lines is disqualified.
The `claude -p` subprocesses are clean raters precisely because they start empty; keep
it that way and never paste the answer key into the same context as the image.

Practical consequences:

* Quote K-series (high-entropy) figures when you care about the worst case: hashes, IDs,
  code, tabular data, anything a language prior cannot repair.
* Quote J/A-series figures when you care about ordinary prose, where redundancy is real
  and does legitimately help - just do not call it a resolution limit.
* Never evaluate with the ground truth in the same context window as the image.

## Grading

Every image has:

* `groundtruth/<name>.txt` - the exact string that was rendered.
* an entry in `ANSWER_KEY.json` with `probes`: a list of `{{id, q, a}}` covering transcription,
  addressed lookup ("the code on line 0421", "cell F17"), counting, arithmetic over what was read,
  and comprehension.

Probes are designed to be gradeable without a judge model: codes use an unambiguous alphabet
(no I/O/0/1 confusions), lines are numbered so position can be addressed, and table/receipt totals
are computed from the same numbers that were drawn.

```bash
python3 probe.py A01                 # questions for matching images
python3 probe.py A01 --answers       # with the key
python3 probe.py --list              # everything, with density stats
```

To run the whole thing end-to-end against the CLI and grade it automatically:

```bash
python3 run_eval.py --model opus --series A_bitmap_ladder --jobs 3
python3 regrade.py                   # re-score stored answers, no new API calls
python3 report_eval.py               # comparison table
```

`run_eval.py` passes `--allowedTools Read` and disallows Bash/Grep/etc: without that, a
model may try to crop and zoom the image with Python rather than reading it, which measures
the wrong thing. It also asks for `UNREADABLE` instead of a guess, so abstention is
distinguishable from error.

## Method notes

* Send PNGs. The JPEG variants exist to measure compression damage - don't let them in by accident.
* Check what your client does before upload: many resize or re-encode, which is exactly the
  failure mode `E_*_09_downup_50pct` and `oversized/` reproduce.
* Ask for verbatim transcription *and* an explicit "which lines could you not read" - a model
  that confabulates plausible text scores far better on fuzzy metrics than it deserves. The
  synthetic corpus is deliberately unguessable: random codes and arbitrary quantities, so
  a hallucinated line is obvious.
* Compare A (bitmap) against B (TrueType) at the same glyph height to separate *glyph size*
  from *anti-aliasing blur*. They diverge sharply below ~8px.

## Regenerating

```bash
python3 generate.py && python3 make_docs.py
```

Content is seeded per image, so a rebuild reproduces byte-identical images and answers.
Edit the ladders (`BITMAPS`, `TTF_SIZES`) or canvas size at the top of the series functions.

*Note: Pillow mis-decodes X11 PCF fonts that declare `first_col > 0` (8x16, 12x24 come out
shifted one glyph); `gen_lib.py` patches the encoding lookup.*
"""
import glob, collections as _c
res = []
for f in sorted(glob.glob(os.path.join(ROOT, 'results_*.json'))):
    b = os.path.basename(f)
    if any(x in b for x in ('confirm', 'edge', 'span')): continue
    blob = json.load(open(f))
    if isinstance(blob, list): res += blob
if res:
    by = {}
    for r in res: by[(r['file'], r['model'])] = r        # later run wins
    models = sorted({m for _, m in by})
    files = sorted({f for f, _ in by},
                   key=lambda k: E[k]['chars_per_claude_token'], reverse=True)
    def c(r):
        if r is None or r.get('code_acc') is None: return '-'
        mis = r.get('misaddressed', 0); ab = r.get('abstained', 0)
        t = f"{r['code_acc']*100:.0f}%"
        if mis: t += f" ({mis} misaddr)"
        if ab:  t += f" ({ab} abstain)"
        return t
    lines = ["", "## Measured results (`claude -p`, series A, 896x896 pages of 148 lines)", "",
             "*exact* = 'the 5-character code on line NNNN' answered exactly.  *legible* = similarity",
             "of a transcribed line to whichever ground-truth line it actually matches best - i.e.",
             "pure glyph legibility with row-addressing errors factored out.  The gap between the two",
             "columns is the model reading the right characters off the wrong row.", "",
             "| glyph | cap px | ch/token | " + ' | '.join(f"{m} exact | {m} legible" for m in models) + " |",
             "|---|---|---|" + "---|" * (2 * len(models))]
    for f in files:
        g = E[f].get('cell', '?')
        row = f"| {g} | {g.split('x')[1] if 'x' in g else '?'} | {E[f]['chars_per_claude_token']:.1f} | "
        for m in models:
            r = by.get((f, m))
            if r is None or r.get('code_acc') is None:
                row += "- | - | "; continue
            mark = '*' if r.get('abstained') else ('+' if r.get('misaddressed') else ' ')
            row += f"{r['code_acc']*100:.0f}%{mark} | {r.get('legibility', 0):.2f} | "
        lines.append(row)
    lines += ["", "`*` abstained (said UNREADABLE) on at least one probe; `+` returned a real code",
              "from an adjacent row.", "",
              "Both models abstain rather than confabulate at and below 5x8, which is the behaviour",
              "you want - the corpus is random, so any invented line would have scored zero anyway.",
              "The interesting band is 6x9-7x14: glyphs are read essentially perfectly (legible",
              ">= 0.96) while exact lookup sits at 25-75%, because on a 148-line page the model",
              "loses track of *which row* it is on. Legibility and addressing fail at different sizes,",
              "and only the addressing failure is silent.", ""]
    readme += chr(10).join(lines) + chr(10)
# ---- pitch / entropy tables from the legibility-only probe
edge_p = os.path.join(ROOT, 'results_edge_opus.json')
if os.path.exists(edge_p):
    ed = json.load(open(edge_p))
    def geo(r):
        v = E[r['file']]
        return v.get('pitch') or int(str(v.get('cell', '0x0')).split('x')[1]), v.get('cell'), v
    js = [r for r in ed if r['file'].startswith(('J_pitch', 'H_'))]
    ks = [r for r in ed if r['file'].startswith('K_entropy')]
    lines = ["", "## What actually limits density", "",
             "Legibility-only probe: transcribe the first 3 and last 2 lines. Those sit at the",
             "top and bottom edges, so no row counting is involved and the score isolates glyph",
             "resolution from addressing.", "",
             "| font | glyph cell | row pitch | ch/token | legibility | confidence |",
             "|---|---|---|---|---|---|"]
    for r in sorted(js, key=lambda r: -r['ch_per_tok']):
        v = E[r['file']]
        lines.append(f"| {v['font'].replace('X11 ', '')} | {v.get('glyph_cell', v.get('cell'))} | "
                     f"{v.get('pitch', '-')} | {r['ch_per_tok']:.1f} | {r['legibility']:.2f} | {r['confidence']} |")
    if ks:
        lines += ["", "### Redundancy credit: same geometry, unguessable payload", "",
                  "| font | pitch | ch/token | template text | random codes | credit |",
                  "|---|---|---|---|---|---|"]
        byk = {}
        for r in js: byk[(E[r['file']].get('font'), E[r['file']].get('pitch'))] = r
        for r in sorted(ks, key=lambda r: -r['ch_per_tok']):
            v = E[r['file']]
            tw = byk.get((v.get('font'), v.get('pitch')))
            t = f"{tw['legibility']:.2f}" if tw else '-'
            d = f"+{tw['legibility'] - r['legibility']:.2f}" if tw else '-'
            lines.append(f"| {v['font'].replace('X11 ', '')} | {v.get('pitch')} | "
                         f"{r['ch_per_tok']:.1f} | {t} | {r['legibility']:.2f} | {d} |")
        lines += ["", "The credit column is how much of the apparent legibility was the language",
                  "model repairing template text rather than resolving pixels.", ""]
    # recommended operating points, derived from the measurements
    def densest(rs, thr=0.95):
        ok = [r for r in rs if r['legibility'] >= thr]
        return max(ok, key=lambda r: r['ch_per_tok']) if ok else None
    k2 = [r for r in ed if r['file'].startswith('K2_')]
    prose = densest(js)
    pool = k2 or ks
    rand = densest(pool)
    rand2 = max([r for r in pool if 0.93 <= r['legibility'] < 0.95],
                key=lambda r: r['ch_per_tok'], default=None)
    lines += ["", "## The answer", "",
              "Headline points require legibility >= 0.95. Everything between 0.80 and 0.94 is a",
              "usable-but-unstable band: `clR6x8` scored 0.80 on one run and 0.00 on an identical",
              "re-run, and the three 5x9 faces scored 0.94 / 0.88 / 0.80. Near the floor a single",
              "run is not a measurement - the boundary is a band, not a line.", ""]
    if prose:
        v = E[prose['file']]
        lines.append(f"* **Redundant text (prose, logs, templated records):** `{v['font'].replace('X11 ','')}` "
                     f"at {v.get('pitch', v.get('cell'))}px pitch - **{prose['ch_per_tok']:.1f} chars per image "
                     f"token** at legibility {prose['legibility']:.2f}. Against measured text "
                     f"tokenisation that is {prose['ch_per_tok']/4.0:.1f}x cheaper for typical English "
                     f"(4.0 chars/token) and {prose['ch_per_tok']/2.17:.1f}x for prose carrying codes "
                     f"and numbers (2.17, measured on this corpus).")
    if rand:
        v = E[rand['file']]
        lines.append(f"* **Unguessable payload, tested ASCII alphabets only:** `{v['font'].replace('X11 ','')}` "
                     f"- **{rand['ch_per_tok']:.1f} chars per image token** at legibility {rand['legibility']:.2f}. "
                     f"Less dense in absolute terms, but random strings tokenise at only 1.14 chars per "
                     f"text token (measured), so this is the bigger saving: "
                     f"{rand['ch_per_tok']/1.14:.1f}x cheaper than sending it as text." if prose else "")
    if rand2:
        v = E[rand2['file']]
        lines.append(f"  Practical alternative: `{v['font'].replace('X11 ','')}` at "
                     f"**{rand2['ch_per_tok']:.1f} chars per image token**, legibility "
                     f"{rand2['legibility']:.3f} - a whisker under the 0.95 bar but "
                     f"{rand2['ch_per_tok']/rand['ch_per_tok']:.1f}x denser, and "
                     f"{rand2['ch_per_tok']/1.14:.1f}x cheaper than text.")
    lines += ["",
              "Pushing past the headline: ~17 chars/token (5x9 cells, 45 px/char) still returns",
              "0.80-0.94 on redundant text, so it is worth trying if you can tolerate re-reads or",
              "verify the output. Below 4x6 glyphs nothing worked at any pitch: 4px-wide glyphs",
              "scored 0.00 at pitch 9 and 10, and only recovered at pitch 12 (15.9 ch/token), by",
              "which point a 5x10 cell is both denser and far more reliable.", "",
              "### One maximum-size image", "",
              "The downscale threshold was located by paired deltas (no baseline assumption:",
              "compare billed tokens between two sizes and check the difference against",
              "`(W2^2-W1^2)/784`). 1568 -> 1904 gave a measured delta of 1489 against a nominal",
              "1488, so **1904x1904 (3.63 MP, 4,624 tokens) passes through untouched**. 1904 ->",
              "2044 gave 202 against a nominal 705, so 2044 is downscaled. The exact cap lies",
              "between them and was not resolved - single runs there sit inside the run-to-run",
              "variation of the fixed prompt overhead.", "",
              "| use | cell | cols x rows | chars in one image | ch/token |",
              "|---|---|---|---|---|",
              "| prose, sim 0.99 | 5x10 | 380 x 190 | **72,200** | 15.6 |",
              "| high-entropy, sim 0.948 | 6x13 | 317 x 146 | 46,282 | 10.0 |",
              "| tested ASCII, sim 0.978 | 8x16 | 238 x 119 | 28,322 | 6.1 |", "",
              "Those are *gross* capacities at similarity scores, not exact-decode guarantees,",
              "and the high-entropy rows use the confusable-free alphabet. Treat them as the",
              "physical ceiling, not as safely retrievable payload.", "",
              "High-entropy payload needs short lines (<=64 chars), but that does not force small",
              "pages - untested, but multiple narrow columns on a large canvas should keep both the",
              "line length and the density. Verify before relying on it.", "",
              "Both assume zero-waste geometry: canvas dimensions divisible by 224 (= lcm(28,32))",
              "and by the glyph cell, and cells actually filled - ragged text at 40% fill throws",
              "away more than the font choice ever wins back.", ""]
    if k2:
        lines += ["### Narrow-page control", "",
                  "K pages carry 224-character lines, so transcribing five of them is a ~660-character",
                  "random-string task - a failure there could be output stamina rather than reading.",
                  "K2 renders the same cells at ~56-character lines:", "",
                  "| cell | ch/token | wide page | narrow page |", "|---|---|---|---|"]
        bywide = {E[r['file']].get('cell'): r for r in ks}
        for r in sorted(k2, key=lambda r: -r['ch_per_tok']):
            v = E[r['file']]
            w = bywide.get(v.get('cell'))
            lines.append(f"| {v.get('cell')} | {r['ch_per_tok']:.1f} | "
                         f"{w['legibility']:.2f} | {r['legibility']:.2f} |" if w else
                         f"| {v.get('cell')} | {r['ch_per_tok']:.1f} | - | {r['legibility']:.2f} |")
        lines.append("")
    readme += chr(10).join(lines) + chr(10)
open(os.path.join(ROOT, 'README.md'), 'w').write(readme)

# ------------------------------------------------ contact sheet
cards = []
for k, v in E.items():
    cards.append(f"""<figure><a href="images/{k}"><img src="images/{k}" loading="lazy"></a>
<figcaption><b>{html.escape(os.path.basename(k))}</b><br>{v['w']}x{v['h']} &middot; {v['font']}<br>
{v['chars']:,} chars &middot; {v['claude_tokens']:,} claude-tok &middot;
<b>{v['chars_per_claude_token']:.1f}</b> ch/tok &middot; {v['claude_compression']:.1f}x</figcaption></figure>""")
open(os.path.join(ROOT, 'index.html'), 'w').write(f"""<!doctype html><meta charset=utf-8>
<title>Dense-text vision benchmark</title>
<style>
body{{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1{{font-size:20px}} main{{display:grid;gap:18px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}}
figure{{margin:0;background:#1c1c1f;border:1px solid #333;border-radius:6px;padding:8px}}
img{{width:100%;image-rendering:pixelated;background:#fff;border-radius:3px}}
figcaption{{font-size:11px;color:#aaa;margin-top:6px;font-family:ui-monospace,monospace}}
</style><h1>Dense-text vision benchmark &mdash; {len(E)} images</h1>
<p style="color:#999">Thumbnails are scaled; click for 1:1. Claude 28px/token, GPT-5.6 32px/token.</p>
<main>{''.join(cards)}</main>""")
print("README.md, index.html written")

# ---- confirmatory pass
import glob as _g, statistics as _st
conf = []
for f in sorted(_g.glob(os.path.join(ROOT, 'results_confirm-v1*.json'))):
    blob = json.load(open(f)); conf += blob['results']
if conf:
    man = json.load(open(sorted(_g.glob(os.path.join(ROOT, 'results_confirm-v1_*.json')))[0]))['manifest']
    L = ["", "## Confirmatory pass (`confirm.py`)", "",
         f"Frozen before running: prompts (sha {man['prompt_decode_sha']} / {man['prompt_bind_sha']}), "
         f"harness (sha {man['harness_sha']}), grader, tool policy (`Read` only), line length "
         f"({man['line_chars']} chars), fresh seed namespace, raw responses retained. "
         f"CLI {man['cli_version']}. **n=2 repetitions** - fewer than the 20 a production gate "
         "would need, and stated as such.", "",
         "Two probes per image: **decode** (first line, anchored at the top edge, no row counting) "
         "and **bind** (a named row, scored by which row actually came back).", "",
         "| payload | alphabet | decode exact | decode CER | bind exact | bind hit a real row |",
         "|---|---|---|---|---|---|"]
    ALPHA = {'prose': 'closed-vocabulary English', 'alnum_easy': 'A-Z2-9, no I/O/0/1',
             'hex': '0-9a-f', 'b64': 'A-Za-z0-9+/ (I, l, O, 0 present)'}
    for pl in ['prose', 'alnum_easy', 'hex', 'b64']:
        dec = [r for r in conf if r['payload'] == pl and r['probe'] == 'decode']
        bnd = [r for r in conf if r['payload'] == pl and r['probe'] == 'bind']
        if not dec: continue
        fr = sum(1 for r in bnd if r['displacement'] is not None)
        L.append(f"| {pl} | {ALPHA[pl]} | **{sum(r['exact'] for r in dec)}/{len(dec)}** | "
                 f"{_st.mean(r['cer'] for r in dec):.3f} | {sum(r['exact'] for r in bnd)}/{len(bnd)} | "
                 f"{fr}/{len(bnd)} |")
    L += ["", "| cell | ch/token | high-entropy decode exact | mean CER | best CER |", "|---|---|---|---|---|"]
    for c in sorted({r['font'] for r in conf},
                    key=lambda c: -next(r['ch_per_token'] for r in conf if r['font'] == c)):
        hi = [r for r in conf if r['font'] == c and r['payload'] != 'prose' and r['probe'] == 'decode']
        if not hi: continue
        L.append(f"| {hi[0]['cell']} | {hi[0]['ch_per_token']:.1f} | **{sum(r['exact'] for r in hi)}/{len(hi)}** | "
                 f"{_st.mean(r['cer'] for r in hi):.3f} | {min(r['cer'] for r in hi):.3f} |")
    L += ["", "### What this overturns", "",
          "Similarity scores of 0.95-0.98 corresponded to an exact-match rate of **zero** on every",
          "high-entropy payload at every cell in the main pass, including the 8x16 row this README",
          "previously called safe. A CER of 0.018 on a 56-character record is one wrong character -",
          "enough for a perfect similarity score and a useless hash.", "",
          "So the useful quantity is not characters per token but",
          "",
          "    rho_operational = (payload chars / image tokens) x P(exact decode AND correct binding)",
          "",
          "and for unguessable data at the densities this benchmark explored, the second factor is 0.",
          "The gross capacity figures remain physically true and are the right number for *semantic*",
          "recall of redundant text, where prose decoded exactly 5/6 at 6x13 and 8x16. They are the",
          "wrong number for anything that must come back byte-exact.", "",
          "Binding fails independently and more often: on the bind probe the returned text frequently",
          "matched no row in the image at all. Decode success does not imply the content came from",
          "the row that was asked for.", ""]
    readme += chr(10).join(L) + chr(10)
    open(os.path.join(ROOT, 'README.md'), 'w').write(readme)

# ---- span-length / alphabet sweep
import glob as _g2, statistics as _st2
spans = []
for f in sorted(_g2.glob(os.path.join(ROOT, 'results_span_*.json'))):
    spans += json.load(open(f))
if spans:
    L = ["", "## Span length and alphabet decide exact recovery, not glyph size", "",
         "The confirmatory pass found CER flat at 0.018-0.036 from 8x16 all the way to 12x24",
         "(288 px/char) - 2.3x more pixels per character bought nothing. So the residual is not",
         "an optical limit. This sweep asks for a short anchored prefix instead of a whole",
         "record, holding the image fixed.", "",
         "| cell | ch/token | alphabet | span 8 | span 16 | span 32 | span 51 |",
         "|---|---|---|---|---|---|---|"]
    for (cell, pl) in sorted({(r['cell'], r['payload']) for r in spans}):
        row = [r for r in spans if r['cell'] == cell and r['payload'] == pl]
        cells = []
        for n in (8, 16, 32, 51):
            rs = [r for r in row if r['span'] == n]
            cells.append(f"{sum(r['exact'] for r in rs)}/{len(rs)}" if rs else '-')
        dens = {'6x13': '10.1', '9x18': '4.8'}.get(cell, '?')
        L.append(f"| {cell} | {dens} | {pl} | " + ' | '.join(f"**{c}**" if c.startswith(('3/3','2/2')) else c
                                                            for c in cells) + " |")
    L += ["", "Two clean effects:", "",
          "1. **Short spans are byte-exact where whole records are not.** hex at 6x13 - 10.1",
          "   chars/token, a density at which full 56-character lines never decoded exactly -",
          "   returns 9/9 exact across spans of 8, 16 and 32 characters. Exactness is bounded by",
          "   how long an exact sequence is requested, not by how small the glyphs are.",
          "2. **Mixed case breaks at any span.** base64 at the same cell is 0/12, failing even at",
          "   8 characters. The errors are case confusions - `SYf`->`SYF`, `6yMF`->`6yMf` - because",
          "   separating f from F needs x-height against cap-height, which small cells destroy.",
          "   Uppercase-only alnum ran at CER 0.071 where mixed-case base64 hit 0.515.", "",
          "The rule is **a prevalidated optical-safe alphabet**, not a cardinality limit: two",
          "alphabets of equal size can have completely different confusion matrices (0/O, 1/I/l,",
          "5/S, 2/Z, B/8, case pairs, stroke density). Lowercase hex is what was tested; nothing",
          "here licenses transferring that result to another 16-symbol set without measuring it.", "",
          "Caveat on this instrument: asking for 'the next N characters' makes the model count to",
          "N, so the probe measures counting as well as reading. Both span-51 failures at 9x18 were",
          "boundary artifacts - one transcribed 50 of 51 characters correctly and missed only the",
          "last, the other abstained - so these numbers understate legibility. A delimited field",
          "('the third 4-character group') would isolate reading properly.", "",
          "### Anchored decode density - NOT operational density", "",
          "The span probe told the model where to look ('the first line'), so it measured",
          "P(exact | correctly anchored). It did not measure finding the right record. The honest",
          "decomposition is:", "",
          "    rho_gross       = chars represented / image tokens",
          "    rho_anchored    = rho_gross x P(exact | correct anchor)",
          "    rho_operational = rho_gross x P(correct anchor) x P(exact | correct anchor)", "",
          "This benchmark measures the first two. **rho_anchored ~= 10 chars/image-token** at 6x13",
          "for short lowercase-hex fields. rho_operational is unidentified and lower, because the",
          "binding probe measured P(correct anchor) as poor: 3/6 for prose and 0/6 for every",
          "high-entropy alphabet, with the returned text often matching no row in the image at all.",
          "Binding, not decoding, is the dominant failure.", "",
          "### Confidence, not point estimates", "",
          "Every rate here comes from small n, so one-sided 95% bounds matter more than the ratio:", "",
          "| observation | bound |",
          "|---|---|",
          "| 0/18 long high-entropy exact (pooled) | P_exact < 0.153 |",
          "| 0/2 per cell/payload condition | P_exact < 0.776 |",
          "| 9/9 short hex spans | P_exact > 0.717 |",
          "| 3/3 at one span length | P_exact > 0.368 |",
          "| 59/59 would be needed | P_exact > 0.95 |",
          "| 299/299 would be needed | P_exact > 0.99 |", "",
          "So the defensible claims are *no tested long high-entropy configuration demonstrated",
          "reliable exact recovery* and *short hex spans are a promising candidate regime* - not",
          "P_exact = 0 and not a reliability guarantee. Pooling across payloads and cells to get",
          "n=18 is itself invalid for estimating one probability; the conditions are heterogeneous.", "",
          "### Literal vs value equality", "",
          "Hex tolerates case-folding and regrouping without changing the value, so exactness was",
          "rescored as `decodeHex(got) == decodeHex(want)` alongside literal string equality. The",
          "two are **identical in every condition**: the hex failures were real digit substitutions",
          "(`377d 3` for `377d 7`), not normalisation artifacts. For base64 case is semantic, so no",
          "such normalisation is permitted there.", ""]
    readme += chr(10).join(L) + chr(10)
    open(os.path.join(ROOT, 'README.md'), 'w').write(readme)

# ---- screening stage (n=20) + geometry
scr = []
for f in sorted(_g2.glob(os.path.join(ROOT, 'results_span_6x13_*.json'))):
    scr += json.load(open(f))
if scr:
    L = ["", "## Screening stage (n=20 per condition)", "",
         "The 3-rep span result was small-sample luck. At n=20 the same condition - lowercase",
         "hex, 32-character span, 6x13 - drops from 3/3 to **15/20**:", "",
         "| alphabet | symbols | span 8 | span 16 | span 32 | span 51 |",
         "|---|---|---|---|---|---|"]
    for pl, card in [('hex', 16), ('crock32', 32)]:
        cells = []
        for n in (8, 16, 32, 51):
            rs = [r for r in scr if r['payload'] == pl and r['span'] == n]
            if rs: cells.append(f"{sum(r['exact'] for r in rs)}/{len(rs)}")
            else: cells.append('-')
        L.append(f"| {pl} | {card} | " + ' | '.join(cells) + " |")
    L += ["", "One-sided 95% Clopper-Pearson lower bounds, hex: span 8 -> 0.717, span 16 -> 0.599,",
          "span 32 -> 0.544. Nothing approaches the 0.95 an unverified handle would need.", "",
          "**A confusion-resistant alphabet lost badly to plain lowercase hex.** Crockford base32",
          "(uppercase, I/L/O/U removed - designed for exactly this problem) scored 4/20 at span 32",
          "against hex's 15/20. At 6px glyph width, alphabet *size* dominates confusion-resistant",
          "*design*: 32 uppercase forms are harder to separate than 16 lowercase-plus-digit forms,",
          "whose ascenders and descenders differ. This is the concrete case for measuring an",
          "alphabet rather than reasoning about it - the a-priori-safer set was 3.75x worse.", "",
          "So the corrected anchored density at 6x13 for 32-character hex fields is", "",
          "    rho_anchored = 10.1 x 0.75 = 7.6 chars/image-token   (95% LB: 5.5)", "",
          "and that still excludes P(correct anchor), which the binding probe put near zero for",
          "high-entropy payloads. **No tested configuration reaches operational reliability**",
          "without a checksum and a canonical fetch.", "",
          "## Maximum canvas, settled", "",
          "Paired deltas, min of 3 reps, overhead ~148 tokens from the 1904 baseline:", "",
          "| canvas | implied image tokens | nominal | ratio |",
          "|---|---|---|---|",
          "| 1904x1904 | 4,623 | 4,624 | 1.000 |",
          "| 1932x1932 | 4,743 | 4,761 | 0.996 |",
          "| 1960x1960 | 4,762 | 4,900 | 0.972 |",
          "| 1456x2576 | 2,981 | 4,784 | 0.623 |", "",
          "**The cap is 69 x 69 = 4,761 patches, so 1932x1932 is the largest square.** 1960x1960",
          "bills an implied 4,762 - it is downscaled to exactly that cap. But the cap is not a",
          "pure patch budget: a 1456x2576 canvas is also 4,784 patches and comes back at 0.623 of",
          "nominal, implying a downscale to roughly 1149x2033. A separate long-edge limit applies,",
          "so the full budget is not reachable by going rectangular.", "",
          "At 1932x1932: 5x10 holds 74,498 chars, 6x13 holds 47,656, 8x16 holds 28,920 - about 3%",
          "over 1904x1904, so either is a fine operating point.", ""]
    readme += chr(10).join(L) + chr(10)
    open(os.path.join(ROOT, 'README.md'), 'w').write(readme)

# ---- error model + corrected taxonomy
if scr:
    L = ["", "## The channel is sparse symbol substitution, not OCR collapse", "",
         "Fitting `P_exact(n) = q^n` jointly across spans 8/16/32 for lowercase hex at 6x13:", "",
         "    q = 0.9890      per-symbol error 1.10%", "",
         "| span | observed | q^n predicted |", "|---|---|---|",
         "| 8 | 18/20 = 0.90 | 0.915 |", "| 16 | 16/20 = 0.80 | 0.838 |",
         "| 32 | 15/20 = 0.75 | 0.702 |", "| 51 | **0/20** | 0.569 |", "",
         "Spans 8-32 fit a single per-symbol rate closely. Span 51 does not: under the fitted",
         "model, `P(0/20) = 4.9e-8`. So the 51-character task is a **separate failure regime** -",
         "counting, boundary tracking, output-span control or abstention - not more of the same",
         "optical noise. (An earlier version of this README claimed the span data refuted a",
         "per-character model; that was based on a 3/3 sample and is withdrawn.)", "",
         "Failures are also sparse: 1.0 character edits per failed record at spans 16 and 32,",
         "2.0 at span 8. Nearly every failure is a single wrong symbol.", "",
         "**Engineering consequence: add detection and correction, not pixels.** Larger cells did",
         "not move CER (0.027 -> 0.018 from 8x16 to 12x24), but a single-symbol-correcting code",
         "over a short field would recover almost every observed failure. The safety metric that",
         "matters is then not `P(literal mismatch)` but", "",
         "    P(a corrupted handle resolves to a DIFFERENT valid record)", "",
         "which must be driven to zero by checksum plus unique-match rejection, never by trusting",
         "the transcription.", "",
         "### Density taxonomy", "",
         "| quantity | 6x13 value | meaning |", "|---|---|---|",
         "| `rho_gross` | 10.1 | characters physically rendered per image token |",
         "| `rho_symbol` | ~10.0 | expected *correct* symbols per image token (CER 0.008) |",
         "| `rho_exact-record` | ~7.6 | all-or-nothing yield at the 32-char success rate 0.75 |",
         "| `rho_operational` | **unknown** | includes binding, checksum rejection, retries, fetch |", "",
         "`rho_exact-record` is a throughput proxy that assumes one correctly anchored record, zero",
         "utility for partial reads, and no cost for retries or canonical fetch. It is not a channel",
         "capacity.", "",
         "### Corrections to earlier claims here", "",
         "* **Crockford tolerant decoding rescues nothing.** Rescoring with case-folding and",
         "  `O->0`, `I/L->1` recovered **0** of the literal failures at every span, so base32's",
         "  result is genuine symbol confusion rather than a normalisation artifact. Hex was",
         "  rescored the same way with the same outcome.",
         "* **The hex-vs-base32 comparison is confounded.** 32 hex characters carry 128 bits;",
         "  32 base32 characters carry 160. Equal character count is not equal information, so",
         "  'alphabet size dominates confusion-resistant design' is not isolated by that test -",
         "  cardinality, case, glyph inventory and bits/symbol all moved together. The delimited",
         "  probe re-runs it at equal bits (32 hex vs 26 base32, 16 vs 13).",
         "* **'15/20 is a floor' is withdrawn.** Removing the counting burden may raise or lower",
         "  performance; changing the task also changes localisation, formatting and abstention.",
         "  15/20 is the measured counted-span result, plausibly conservative for delimited fields,",
         "  not a bound on them.",
         "* **Binding is not 'near zero'.** 0/6 per condition gives a one-sided 95% upper bound of",
         "  0.393, and pooling heterogeneous payloads to n=18 is invalid for one probability. What",
         "  is established is that *dense row-number addressing is a bad protocol* - not that",
         "  binding is an intrinsic optical limit. Hierarchical page/block/handle addressing is",
         "  untested.",
         "* **1932x1932 is strongly supported, not confirmed to the pixel.** The implied-token",
         "  figures subtract an overhead estimated at ~148, and that overhead is not constant",
         "  across runs (147 at 1904, 130 at 1932). The adjacent 1932 vs 1933 test settles it.",
         "* **Rectangular geometry is not refuted in general.** One extreme 1456x2576 canvas failed",
         "  to reach its nominal patch budget; moderate rectangles below the effective long-edge",
         "  limit are unmeasured. The admissible region is undefined, and all geometry probes here",
         "  ran on the Sonnet path - Opus and other endpoints are not assumed identical.", ""]
    readme += chr(10).join(L) + chr(10)
    open(os.path.join(ROOT, 'README.md'), 'w').write(readme)
