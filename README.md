# Dense-text vision benchmark

128 images for probing how small text can get before a vision LLM stops reading it,
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

| font | glyph px | chars in image | Claude img-tokens | chars / Claude token | chars / GPT token | vs. plain-text tokens |
|---|---|---|---|---|---|---|
| 4x6 | 6 | 32,093 | 1,024 | 31.3 | 40.9 | 7.8x |
| 5x7 | 7 | 21,642 | 1,024 | 21.1 | 27.6 | 5.3x |
| 5x8 | 8 | 19,080 | 1,024 | 18.6 | 24.3 | 4.7x |
| 6x9 | 9 | 14,016 | 1,024 | 13.7 | 17.9 | 3.4x |
| 6x10 | 10 | 12,575 | 1,024 | 12.3 | 16.0 | 3.1x |
| 6x12 | 12 | 10,580 | 1,024 | 10.3 | 13.5 | 2.6x |
| 6x13 | 13 | 9,731 | 1,024 | 9.5 | 12.4 | 2.4x |
| 7x13 | 13 | 8,200 | 1,024 | 8.0 | 10.5 | 2.0x |
| 7x14 | 14 | 7,619 | 1,024 | 7.4 | 9.7 | 1.9x |
| 8x13 | 13 | 7,172 | 1,024 | 7.0 | 9.2 | 1.8x |
| 8x16 | 16 | 5,808 | 1,024 | 5.7 | 7.4 | 1.4x |
| 9x15 | 15 | 5,469 | 1,024 | 5.3 | 7.0 | 1.3x |
| 9x18 | 18 | 4,588 | 1,024 | 4.5 | 5.8 | 1.1x |
| 10x20 | 20 | 3,622 | 1,024 | 3.5 | 4.6 | 0.9x |
| 12x24 | 24 | 2,548 | 1,024 | 2.5 | 3.2 | 0.6x |

### B - anti-aliased TrueType mono (DejaVu Sans Mono), 896x896

| font | glyph px | chars in image | Claude img-tokens | chars / Claude token | chars / GPT token | vs. plain-text tokens |
|---|---|---|---|---|---|---|
| 5px | 5 | 36,281 | 1,024 | 35.4 | 46.3 | 8.9x |
| 6px | 6 | 26,389 | 1,024 | 25.8 | 33.7 | 6.4x |
| 7px | 7 | 20,094 | 1,024 | 19.6 | 25.6 | 4.9x |
| 8px | 8 | 15,705 | 1,024 | 15.3 | 20.0 | 3.8x |
| 9px | 9 | 11,471 | 1,024 | 11.2 | 14.6 | 2.8x |
| 10px | 10 | 9,646 | 1,024 | 9.4 | 12.3 | 2.4x |
| 11px | 11 | 8,066 | 1,024 | 7.9 | 10.3 | 2.0x |
| 12px | 12 | 6,888 | 1,024 | 6.7 | 8.8 | 1.7x |
| 14px | 14 | 5,172 | 1,024 | 5.0 | 6.6 | 1.3x |
| 16px | 16 | 3,947 | 1,024 | 3.9 | 5.0 | 1.0x |
| 20px | 20 | 2,443 | 1,024 | 2.4 | 3.1 | 0.6x |
| 24px | 24 | 1,686 | 1,024 | 1.6 | 2.1 | 0.4x |
| 32px | 32 | 942 | 1,024 | 0.9 | 1.2 | 0.2x |

### C - proportional prose (DejaVu Sans), 896x896

| font | glyph px | chars in image | Claude img-tokens | chars / Claude token | chars / GPT token | vs. plain-text tokens |
|---|---|---|---|---|---|---|
| 5px | 5 | 37,854 | 1,024 | 37.0 | 48.3 | 9.2x |
| 6px | 6 | 27,954 | 1,024 | 27.3 | 35.7 | 6.8x |
| 7px | 7 | 21,203 | 1,024 | 20.7 | 27.0 | 5.2x |
| 8px | 8 | 16,900 | 1,024 | 16.5 | 21.6 | 4.1x |
| 9px | 9 | 12,597 | 1,024 | 12.3 | 16.1 | 3.1x |
| 10px | 10 | 10,467 | 1,024 | 10.2 | 13.3 | 2.6x |
| 12px | 12 | 7,660 | 1,024 | 7.5 | 9.8 | 1.9x |
| 14px | 14 | 5,881 | 1,024 | 5.7 | 7.5 | 1.4x |
| 16px | 16 | 4,729 | 1,024 | 4.6 | 6.0 | 1.1x |
| 20px | 20 | 2,993 | 1,024 | 2.9 | 3.8 | 0.7x |
| 24px | 24 | 2,038 | 1,024 | 2.0 | 2.6 | 0.5x |

The last column is the interesting one: at 4x6 a page holds ~8,000 plain-text tokens' worth
of characters in 1,024 image tokens. If a model can actually read it, images are an 8x cheaper
channel than text - and the ladder tells you exactly where that stops being true.

## Series

| dir | n | what it isolates |
|---|---|---|
| `A_bitmap_ladder` | 15 | glyph size, with anti-aliasing removed as a variable. 4x6 up to 12x24. |
| `B_mono_ttf_ladder` | 18 | the same ladder in anti-aliased TrueType, 5px..32px, plus 1-bit `noAA` twins at 6/8/10/12/16px. |
| `C_prose_proportional` | 11 | proportional prose - real reading rather than code-lookup, and higher chars/px than mono. |
| `D_eyecharts` | 3 | every size in a single image; finds the floor in one request. |
| `E_degradations` | 28 | contrast, inversion, colour, blur, noise, 1.5deg rotation, resampling, JPEG q30/q60 - and two oversized pages the provider will downscale for you. |
| `F_layouts` | 15 | tables with column totals, code listings, 3-column newsprint, narrow receipts, dark terminal logs, spreadsheet grids with A1-style addressing. |
| `G_multiscale` | 2 | one page cascading from a 30px title to 4x6 micro-print, each tier carrying its own code - shows where reading fails *within* one image. |

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
* an entry in `ANSWER_KEY.json` with `probes`: a list of `{id, q, a}` covering transcription,
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

## Measured results (`claude -p`, series A, 896x896 pages of 148 lines)

*exact* = 'the 5-character code on line NNNN' answered exactly.  *legible* = similarity
of a transcribed line to whichever ground-truth line it actually matches best - i.e.
pure glyph legibility with row-addressing errors factored out.  The gap between the two
columns is the model reading the right characters off the wrong row.

| glyph | cap px | ch/token | opus exact | opus legible | sonnet exact | sonnet legible |
|---|---|---|---|---|---|---|
| 4x6 | 6 | 31.3 | 0%* | 0.00 | 0%* | 0.00 | 
| 5x6 | 6 | 25.6 | 0%* | 0.00 | - | - | 
| 6x6 | 6 | 21.3 | 0%* | 0.00 | - | - | 
| 5x7 | 7 | 21.1 | 0%* | 0.00 | 0%* | 0.00 | 
| 5x8 | 8 | 19.2 | 0%* | 0.00 | - | - | 
| 5x8 | 8 | 18.6 | 0%* | 0.00 | 0%* | 0.00 | 
| 6x8 | 8 | 15.6 | 0%* | 0.00 | - | - | 
| 5x10 | 10 | 15.2 | 50%  | 0.99 | - | - | 
| 6x9 | 9 | 14.1 | 50%+ | 0.96 | - | - | 
| 6x9 | 9 | 14.0 | 0%+ | 0.93 | - | - | 
| 6x9 | 9 | 14.0 | 25%  | 0.97 | - | - | 
| 6x9 | 9 | 13.8 | 0%* | 0.96 | - | - | 
| 6x9 | 9 | 13.7 | 25%+ | 0.99 | 25%* | 0.00 | 
| 7x8 | 8 | 13.5 | 0%* | 0.00 | - | - | 
| 6x10 | 10 | 12.5 | 0%  | 0.95 | - | - | 
| 6x10 | 10 | 12.3 | 50%  | 1.00 | 25%* | 0.89 | 
| 8x8 | 8 | 11.7 | 0%* | 0.00 | - | - | 
| 6x12 | 12 | 10.3 | 50%  | 0.98 | 75%  | 0.78 | 
| 6x13 | 13 | 9.5 | 75%  | 1.00 | 100%+ | 0.94 | 
| 7x13 | 13 | 8.0 | 75%+ | 0.96 | 100%  | 0.96 | 
| 7x14 | 14 | 7.4 | 75%  | 1.00 | 75%  | 0.85 | 
| 8x13 | 13 | 7.0 | 100%  | 0.97 | 100%  | 0.95 | 
| 8x16 | 16 | 5.7 | 75%  | 1.00 | 100%  | 0.95 | 
| 9x15 | 15 | 5.3 | 100%  | 1.00 | 50%  | 0.94 | 
| 9x18 | 18 | 4.5 | 100%  | 1.00 | 100%  | 0.92 | 
| 10x20 | 20 | 3.5 | 100%  | 1.00 | 75%+ | 0.94 | 
| 12x24 | 24 | 2.5 | 100%  | 1.00 | 100%  | 1.00 | 

`*` abstained (said UNREADABLE) on at least one probe; `+` returned a real code
from an adjacent row.

Both models abstain rather than confabulate at and below 5x8, which is the behaviour
you want - the corpus is random, so any invented line would have scored zero anyway.
The interesting band is 6x9-7x14: glyphs are read essentially perfectly (legible
>= 0.96) while exact lookup sits at 25-75%, because on a 148-line page the model
loses track of *which row* it is on. Legibility and addressing fail at different sizes,
and only the addressing failure is silent.


## What actually limits density

Legibility-only probe: transcribe the first 3 and last 2 lines. Those sit at the
top and bottom edges, so no row counting is involved and the score isolates glyph
resolution from addressing.

| font | glyph cell | row pitch | ch/token | legibility | confidence |
|---|---|---|---|---|---|
| 4x6 | 4x6 | 9 | 21.3 | 0.00 | low |
| 4x6 | 4x6 | 10 | 19.1 | 0.00 | low |
| clR5x8 | 5x8 | 9 | 17.0 | 0.94 | low |
| clR5x6 | 5x6 | 9 | 16.9 | 0.80 | low |
| 5x7 | 5x7 | 9 | 16.9 | 0.88 | low |
| 4x6 | 4x6 | 12 | 15.9 | 0.90 | low |
| clR6x8 | 6x8 | - | 15.6 | 0.00 | low |
| clR5x10 | 5x10 | - | 15.2 | 0.99 | medium |
| clR5x8 | 5x8 | 10 | 15.2 | 0.99 | medium |
| clR6x8 | 6x8 | 9 | 14.1 | 0.84 | medium |

### Redundancy credit: same geometry, unguessable payload

| font | pitch | ch/token | template text | random codes | credit |
|---|---|---|---|---|---|
| 4x6 | 9 | 21.5 | 0.00 | 0.00 | +0.00 |
| clR5x8 | 9 | 17.2 | 0.94 | 0.00 | +0.94 |
| clR6x8 | 8 | 15.9 | - | 0.00 | - |
| 6x9 | 9 | 14.2 | - | 0.00 | - |
| 6x10 | 10 | 12.7 | - | 0.00 | - |
| 6x13 | 13 | 9.8 | - | 0.59 | - |
| 7x14 | 14 | 8.0 | - | 0.69 | - |
| 8x16 | 16 | 5.9 | - | 0.96 | - |
| 9x18 | 18 | 4.7 | - | 1.00 | - |
| 10x20 | 20 | 3.8 | - | 0.99 | - |
| 12x24 | 24 | 2.6 | - | 1.00 | - |

The credit column is how much of the apparent legibility was the language
model repairing template text rather than resolving pixels.


## The answer

Headline points require legibility >= 0.95. Everything between 0.80 and 0.94 is a
usable-but-unstable band: `clR6x8` scored 0.80 on one run and 0.00 on an identical
re-run, and the three 5x9 faces scored 0.94 / 0.88 / 0.80. Near the floor a single
run is not a measurement - the boundary is a band, not a line.

* **Redundant text (prose, logs, templated records):** `clR5x10` at 5x10px pitch - **15.2 chars per image token** at legibility 0.99. Against measured text tokenisation that is 3.8x cheaper for typical English (4.0 chars/token) and 7.0x for prose carrying codes and numbers (2.17, measured on this corpus).
* **Unguessable payload, tested ASCII alphabets only:** `8x16` - **5.9 chars per image token** at legibility 0.98. Less dense in absolute terms, but random strings tokenise at only 1.14 chars per text token (measured), so this is the bigger saving: 5.2x cheaper than sending it as text.
  Practical alternative: `6x13` at **9.7 chars per image token**, legibility 0.948 - a whisker under the 0.95 bar but 1.6x denser, and 8.5x cheaper than text.

Pushing past the headline: ~17 chars/token (5x9 cells, 45 px/char) still returns
0.80-0.94 on redundant text, so it is worth trying if you can tolerate re-reads or
verify the output. Below 4x6 glyphs nothing worked at any pitch: 4px-wide glyphs
scored 0.00 at pitch 9 and 10, and only recovered at pitch 12 (15.9 ch/token), by
which point a 5x10 cell is both denser and far more reliable.

### One maximum-size image

The downscale threshold was located by paired deltas (no baseline assumption:
compare billed tokens between two sizes and check the difference against
`(W2^2-W1^2)/784`). 1568 -> 1904 gave a measured delta of 1489 against a nominal
1488, so **1904x1904 (3.63 MP, 4,624 tokens) passes through untouched**. 1904 ->
2044 gave 202 against a nominal 705, so 2044 is downscaled. The exact cap lies
between them and was not resolved - single runs there sit inside the run-to-run
variation of the fixed prompt overhead.

| use | cell | cols x rows | chars in one image | ch/token |
|---|---|---|---|---|
| prose, sim 0.99 | 5x10 | 380 x 190 | **72,200** | 15.6 |
| high-entropy, sim 0.948 | 6x13 | 317 x 146 | 46,282 | 10.0 |
| tested ASCII, sim 0.978 | 8x16 | 238 x 119 | 28,322 | 6.1 |

Those are *gross* capacities at similarity scores, not exact-decode guarantees,
and the high-entropy rows use the confusable-free alphabet. Treat them as the
physical ceiling, not as safely retrievable payload.

High-entropy payload needs short lines (<=64 chars), but that does not force small
pages - untested, but multiple narrow columns on a large canvas should keep both the
line length and the density. Verify before relying on it.

Both assume zero-waste geometry: canvas dimensions divisible by 224 (= lcm(28,32))
and by the glyph cell, and cells actually filled - ragged text at 40% fill throws
away more than the font choice ever wins back.

### Narrow-page control

K pages carry 224-character lines, so transcribing five of them is a ~660-character
random-string task - a failure there could be output stamina rather than reading.
K2 renders the same cells at ~56-character lines:

| cell | ch/token | wide page | narrow page |
|---|---|---|---|
| 6x9 | 14.0 | 0.00 | 0.55 |
| 6x10 | 12.6 | 0.00 | 0.70 |
| 6x13 | 9.7 | 0.59 | 0.95 |
| 8x16 | 5.9 | 0.96 | 0.98 |
| 9x18 | 4.6 | 1.00 | 1.00 |


## Confirmatory pass (`confirm.py`)

Frozen before running: prompts (sha 7ec0b7be25263cad / 3610002e12ac3ba4), harness (sha 89678907c6e937d8), grader, tool policy (`Read` only), line length (56 chars), fresh seed namespace, raw responses retained. CLI 2.1.234 (Claude Code). **n=2 repetitions** - fewer than the 20 a production gate would need, and stated as such.

Two probes per image: **decode** (first line, anchored at the top edge, no row counting) and **bind** (a named row, scored by which row actually came back).

| payload | alphabet | decode exact | decode CER | bind exact | bind hit a real row |
|---|---|---|---|---|---|
| prose | closed-vocabulary English | **5/6** | 0.003 | 3/6 | 4/6 |
| alnum_easy | A-Z2-9, no I/O/0/1 | **0/6** | 0.071 | 0/6 | 1/6 |
| hex | 0-9a-f | **1/12** | 0.039 | 1/12 | 9/12 |
| b64 | A-Za-z0-9+/ (I, l, O, 0 present) | **0/12** | 0.289 | 0/12 | 4/12 |

| cell | ch/token | high-entropy decode exact | mean CER | best CER |
|---|---|---|---|---|
| 5x10 | 15.7 | **0/6** | 0.375 | 0.054 |
| 6x13 | 10.1 | **0/6** | 0.065 | 0.036 |
| 8x16 | 6.1 | **0/6** | 0.193 | 0.018 |
| 9x18 | 4.8 | **0/4** | 0.067 | 0.036 |
| 10x20 | 3.9 | **0/4** | 0.045 | 0.036 |
| 12x24 | 2.7 | **1/4** | 0.027 | 0.000 |

### What this overturns

Similarity scores of 0.95-0.98 corresponded to an exact-match rate of **zero** on every
high-entropy payload at every cell in the main pass, including the 8x16 row this README
previously called safe. A CER of 0.018 on a 56-character record is one wrong character -
enough for a perfect similarity score and a useless hash.

So the useful quantity is not characters per token but

    rho_operational = (payload chars / image tokens) x P(exact decode AND correct binding)

and for unguessable data at the densities this benchmark explored, the second factor is 0.
The gross capacity figures remain physically true and are the right number for *semantic*
recall of redundant text, where prose decoded exactly 5/6 at 6x13 and 8x16. They are the
wrong number for anything that must come back byte-exact.

Binding fails independently and more often: on the bind probe the returned text frequently
matched no row in the image at all. Decode success does not imply the content came from
the row that was asked for.


## Span length and alphabet decide exact recovery, not glyph size

The confirmatory pass found CER flat at 0.018-0.036 from 8x16 all the way to 12x24
(288 px/char) - 2.3x more pixels per character bought nothing. So the residual is not
an optical limit. This sweep asks for a short anchored prefix instead of a whole
record, holding the image fixed.

| cell | ch/token | alphabet | span 8 | span 16 | span 32 | span 51 |
|---|---|---|---|---|---|---|
| 6x13 | 10.1 | b64 | 0/3 | 0/3 | 0/3 | 0/3 |
| 6x13 | 10.1 | crock32 | 9/20 | 7/20 | 4/20 | 0/20 |
| 6x13 | 10.1 | hex | 18/20 | 16/20 | 15/20 | 0/20 |
| 9x18 | 4.8 | hex | **3/3** | **3/3** | **3/3** | 1/3 |

Two clean effects:

1. **Short spans are byte-exact where whole records are not.** hex at 6x13 - 10.1
   chars/token, a density at which full 56-character lines never decoded exactly -
   returns 9/9 exact across spans of 8, 16 and 32 characters. Exactness is bounded by
   how long an exact sequence is requested, not by how small the glyphs are.
2. **Mixed case breaks at any span.** base64 at the same cell is 0/12, failing even at
   8 characters. The errors are case confusions - `SYf`->`SYF`, `6yMF`->`6yMf` - because
   separating f from F needs x-height against cap-height, which small cells destroy.
   Uppercase-only alnum ran at CER 0.071 where mixed-case base64 hit 0.515.

The rule is **a prevalidated optical-safe alphabet**, not a cardinality limit: two
alphabets of equal size can have completely different confusion matrices (0/O, 1/I/l,
5/S, 2/Z, B/8, case pairs, stroke density). Lowercase hex is what was tested; nothing
here licenses transferring that result to another 16-symbol set without measuring it.

Caveat on this instrument: asking for 'the next N characters' makes the model count to
N, so the probe measures counting as well as reading. Both span-51 failures at 9x18 were
boundary artifacts - one transcribed 50 of 51 characters correctly and missed only the
last, the other abstained - so these numbers understate legibility. A delimited field
('the third 4-character group') would isolate reading properly.

### Anchored decode density - NOT operational density

The span probe told the model where to look ('the first line'), so it measured
P(exact | correctly anchored). It did not measure finding the right record. The honest
decomposition is:

    rho_gross       = chars represented / image tokens
    rho_anchored    = rho_gross x P(exact | correct anchor)
    rho_operational = rho_gross x P(correct anchor) x P(exact | correct anchor)

This benchmark measures the first two. **rho_anchored ~= 10 chars/image-token** at 6x13
for short lowercase-hex fields. rho_operational is unidentified and lower, because the
binding probe measured P(correct anchor) as poor: 3/6 for prose and 0/6 for every
high-entropy alphabet, with the returned text often matching no row in the image at all.
Binding, not decoding, is the dominant failure.

### Confidence, not point estimates

Every rate here comes from small n, so one-sided 95% bounds matter more than the ratio:

| observation | bound |
|---|---|
| 0/18 long high-entropy exact (pooled) | P_exact < 0.153 |
| 0/2 per cell/payload condition | P_exact < 0.776 |
| 9/9 short hex spans | P_exact > 0.717 |
| 3/3 at one span length | P_exact > 0.368 |
| 59/59 would be needed | P_exact > 0.95 |
| 299/299 would be needed | P_exact > 0.99 |

So the defensible claims are *no tested long high-entropy configuration demonstrated
reliable exact recovery* and *short hex spans are a promising candidate regime* - not
P_exact = 0 and not a reliability guarantee. Pooling across payloads and cells to get
n=18 is itself invalid for estimating one probability; the conditions are heterogeneous.

### Literal vs value equality

Hex tolerates case-folding and regrouping without changing the value, so exactness was
rescored as `decodeHex(got) == decodeHex(want)` alongside literal string equality. The
two are **identical in every condition**: the hex failures were real digit substitutions
(`377d 3` for `377d 7`), not normalisation artifacts. For base64 case is semantic, so no
such normalisation is permitted there.


## Screening stage (n=20 per condition)

The 3-rep span result was small-sample luck. At n=20 the same condition - lowercase
hex, 32-character span, 6x13 - drops from 3/3 to **15/20**:

| alphabet | symbols | span 8 | span 16 | span 32 | span 51 |
|---|---|---|---|---|---|
| hex | 16 | 18/20 | 16/20 | 15/20 | 0/20 |
| crock32 | 32 | 9/20 | 7/20 | 4/20 | 0/20 |

One-sided 95% Clopper-Pearson lower bounds, hex: span 8 -> 0.717, span 16 -> 0.599,
span 32 -> 0.544. Nothing approaches the 0.95 an unverified handle would need.

**A confusion-resistant alphabet lost badly to plain lowercase hex.** Crockford base32
(uppercase, I/L/O/U removed - designed for exactly this problem) scored 4/20 at span 32
against hex's 15/20. At 6px glyph width, alphabet *size* dominates confusion-resistant
*design*: 32 uppercase forms are harder to separate than 16 lowercase-plus-digit forms,
whose ascenders and descenders differ. This is the concrete case for measuring an
alphabet rather than reasoning about it - the a-priori-safer set was 3.75x worse.

So the corrected anchored density at 6x13 for 32-character hex fields is

    rho_anchored = 10.1 x 0.75 = 7.6 chars/image-token   (95% LB: 5.5)

and that still excludes P(correct anchor), which the binding probe put near zero for
high-entropy payloads. **No tested configuration reaches operational reliability**
without a checksum and a canonical fetch.

## Maximum canvas, settled

Paired deltas, min of 3 reps, overhead ~148 tokens from the 1904 baseline:

| canvas | implied image tokens | nominal | ratio |
|---|---|---|---|
| 1904x1904 | 4,623 | 4,624 | 1.000 |
| 1932x1932 | 4,743 | 4,761 | 0.996 |
| 1960x1960 | 4,762 | 4,900 | 0.972 |
| 1456x2576 | 2,981 | 4,784 | 0.623 |

**The cap is 69 x 69 = 4,761 patches, so 1932x1932 is the largest square.** 1960x1960
bills an implied 4,762 - it is downscaled to exactly that cap. But the cap is not a
pure patch budget: a 1456x2576 canvas is also 4,784 patches and comes back at 0.623 of
nominal, implying a downscale to roughly 1149x2033. A separate long-edge limit applies,
so the full budget is not reachable by going rectangular.

At 1932x1932: 5x10 holds 74,498 chars, 6x13 holds 47,656, 8x16 holds 28,920 - about 3%
over 1904x1904, so either is a fine operating point.


## The channel is sparse symbol substitution, not OCR collapse

Fitting `P_exact(n) = q^n` jointly across spans 8/16/32 for lowercase hex at 6x13:

    q = 0.9890      per-symbol error 1.10%

| span | observed | q^n predicted |
|---|---|---|
| 8 | 18/20 = 0.90 | 0.915 |
| 16 | 16/20 = 0.80 | 0.838 |
| 32 | 15/20 = 0.75 | 0.702 |
| 51 | **0/20** | 0.569 |

Spans 8-32 fit a single per-symbol rate closely. Span 51 does not: under the fitted
model, `P(0/20) = 4.9e-8`. So the 51-character task is a **separate failure regime** -
counting, boundary tracking, output-span control or abstention - not more of the same
optical noise. (An earlier version of this README claimed the span data refuted a
per-character model; that was based on a 3/3 sample and is withdrawn.)

Failures are also sparse: 1.0 character edits per failed record at spans 16 and 32,
2.0 at span 8. Nearly every failure is a single wrong symbol.

**Engineering consequence: add detection and correction, not pixels.** Larger cells did
not move CER (0.027 -> 0.018 from 8x16 to 12x24), but a single-symbol-correcting code
over a short field would recover almost every observed failure. The safety metric that
matters is then not `P(literal mismatch)` but

    P(a corrupted handle resolves to a DIFFERENT valid record)

which must be driven to zero by checksum plus unique-match rejection, never by trusting
the transcription.

### Density taxonomy

| quantity | 6x13 value | meaning |
|---|---|---|
| `rho_gross` | 10.1 | characters physically rendered per image token |
| `rho_symbol` | ~10.0 | expected *correct* symbols per image token (CER 0.008) |
| `rho_exact-record` | ~7.6 | all-or-nothing yield at the 32-char success rate 0.75 |
| `rho_operational` | **unknown** | includes binding, checksum rejection, retries, fetch |

`rho_exact-record` is a throughput proxy that assumes one correctly anchored record, zero
utility for partial reads, and no cost for retries or canonical fetch. It is not a channel
capacity.

### Corrections to earlier claims here

* **Crockford tolerant decoding rescues nothing.** Rescoring with case-folding and
  `O->0`, `I/L->1` recovered **0** of the literal failures at every span, so base32's
  result is genuine symbol confusion rather than a normalisation artifact. Hex was
  rescored the same way with the same outcome.
* **The hex-vs-base32 comparison is confounded.** 32 hex characters carry 128 bits;
  32 base32 characters carry 160. Equal character count is not equal information, so
  'alphabet size dominates confusion-resistant design' is not isolated by that test -
  cardinality, case, glyph inventory and bits/symbol all moved together. The delimited
  probe re-runs it at equal bits (32 hex vs 26 base32, 16 vs 13).
* **'15/20 is a floor' is withdrawn.** Removing the counting burden may raise or lower
  performance; changing the task also changes localisation, formatting and abstention.
  15/20 is the measured counted-span result, plausibly conservative for delimited fields,
  not a bound on them.
* **Binding is not 'near zero'.** 0/6 per condition gives a one-sided 95% upper bound of
  0.393, and pooling heterogeneous payloads to n=18 is invalid for one probability. What
  is established is that *dense row-number addressing is a bad protocol* - not that
  binding is an intrinsic optical limit. Hierarchical page/block/handle addressing is
  untested.
* **1932x1932 is strongly supported, not confirmed to the pixel.** The implied-token
  figures subtract an overhead estimated at ~148, and that overhead is not constant
  across runs (147 at 1904, 130 at 1932). The adjacent 1932 vs 1933 test settles it.
* **Rectangular geometry is not refuted in general.** One extreme 1456x2576 canvas failed
  to reach its nominal patch budget; moderate rectangles below the effective long-edge
  limit are unmeasured. The admissible region is undefined, and all geometry probes here
  ran on the Sonnet path - Opus and other endpoints are not assumed identical.

