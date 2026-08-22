#!/usr/bin/env python3
"""Build README.md + index.html.  Ordered by topic, with corrections folded into the
claims they correct - not appended after them."""
import difflib, json, os, sys, glob, html, math, collections, statistics as st, tempfile, shutil
import provenance as V
from analyze_encoding import paired_difference_interval

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECK = '--check' in sys.argv
OUTPUT_ROOT = tempfile.mkdtemp(prefix='vision-text-docs-') if CHECK else ROOT

def write_output(name, text):
    with open(os.path.join(OUTPUT_ROOT, name), 'w') as f:
        f.write(text)
K = json.load(open(os.path.join(ROOT, 'ANSWER_KEY.json')))
E = K['images']

def load(pat, key=None):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        b = json.load(open(f))
        out += b[key] if key and isinstance(b, dict) else V.result_rows(b)
    return out

edge    = load('results_edge_opus.json')
confirm = load('results_confirm-v1*.json', 'results')
span    = {os.path.basename(f): V.result_rows(json.load(open(f)))
           for f in sorted(glob.glob(os.path.join(ROOT, 'results_span_*.json')))}
delim   = load('results_delim_6x13_opus.json')
ladder  = load('results_opus.json') + load('results_sonnet.json')

def lb(k, n):
    try:
        from scipy.stats import beta
        return beta.ppf(0.05, k, n - k + 1) if k > 0 else 0.0
    except ImportError:
        return float('nan')

def _count_runs():
    n = 0
    for f in glob.glob(os.path.join(ROOT, 'results_*.json')):
        try: b = json.load(open(f))
        except Exception: continue
        if isinstance(b, list): n += len(b)
        elif isinstance(b, dict):
            n += len(b.get('results') or b.get('rows') or [])
    return n
TOTAL_RUNS = _count_runs()

S = []
def w(*lines): S.extend(lines)

# ─────────────────────────────────────────────────────────────── header
w(f"""# Dense text in images

How many characters fit in one image token, and how many of them come back correctly -
measured for **Claude** (28x28 px patches) and the **GPT-5.6 / Sol line** (32x32 px patches).
{len(E)} generated test images, {TOTAL_RUNS:,} recorded model results, and a
record of which conclusions survived scrutiny.

Two separate questions, with very different answers:

* **Capacity** is solved and well measured. A 1932x1932 page holds 74,498 characters of
  prose at 15.7 characters per image token.
* **Fidelity** is not. Arbitrary data comes back byte-exact only in short delimited
  fields, and only some of the time. Nothing measured here is safe to trust unverified.

| | |
|---|---|
| Pixels per image token | Claude **784** (28x28 @ 1 tok/patch); GPT-5.6 **853** (32x32 @ 1.2 tok/patch) |
| Largest un-downscaled square | Claude **1932x1932** (4,761 patches); GPT-5.6 **1600x1600** (2,500 patches) |
| Prose capacity, one image | Claude **74,498 chars** (15.7 ch/tok); GPT-5.6 **51,200** (17.1 ch/tok) |
| Densest cell read at >= 0.95 legibility | Claude 5x10 @ **15.7**; gpt-5.6-sol 5x9 @ **18.4** ch/token |
| Reads dense glyphs best | **gpt-5.6-sol** (mean legibility 0.761 vs opus 0.606, p = 0.0094) |
| Recovers an exact handle best | **claude** (delimited fields 70/80 vs sol 53/80, luna 22/80) |
| Resolves the right record best | **not established** (Claude leads a debug-only n=20 preflight; only sol has Stage 1) |
| Exact recovery of a 56-char record | Claude **0/18**; gpt-5.6 2-4/18 (underpowered, CIs overlap) |
| Biggest safety gap | sol returns a record for **~37%** of Stage-1 no-answer calls; Claude has only 0/5 preflight evidence |

---

## 1. Token geometry

Marginal billed input tokens for the same prompt across image sizes
(`claude -p --output-format json`, deltas so no baseline is assumed):

| image | pixels | billed input | marginal px/token |
|---|---|---|---|
| 224 x 224 | 50,176 | 227 | - |
| 448 x 448 | 200,704 | 392 | 912 |
| 896 x 896 | 802,816 | 1,174 | **770** |
| 1344 x 1344 | 1,806,336 | 2,450 | **786** |

The marginal rate converges on 784 = 28 x 28. So

    tokens = ceil(W/28) x ceil(H/28)
    chars  = floor(W/cw) x floor(H/ch)
    chars per token = 784 / (cw x ch)     when both divide exactly

Density is therefore **independent of aspect ratio and of page size**. Page shape only
affects remainder waste, cell fill, line length and how many rows the model must track -
all of which turn out to matter for fidelity, none for raw density.

**Zero-waste alignment** needs `W = 0 mod lcm(28, cw)` and `H = 0 mod lcm(28, ch)`. A
multiple of 224 = lcm(28, 32) is required only if you also want GPT-5.6's 32px grid to
divide evenly. For a 5x10 cell, lcm(28,5) = lcm(28,10) = 140, so 1540x1540 is exactly
aligned and reaches 784/50 = 15.68 chars/token. Maximum chars/token and maximum
chars/image are different objectives and select different canvases.

### The downscale ceiling

Paired deltas, five reps each, compared against 1904:

| canvas | mean billed | sd | delta vs 1904 | nominal delta | grid |
|---|---|---|---|---|---|
| 1904 x 1904 | 4765.4 | 13.8 | - | - | 68 x 68 = 4,624 |
| 1932 x 1932 | 4899.2 | 7.3 | **+133.8** | +137 | 69 x 69 = 4,761 |
| 1933 x 1933 | 4907.6 | 9.5 | +142.2 | +276 | 70 x 70 = 4,900 |

1933 costs 8.4 tokens more than 1932; uncapped it would cost 139 more. One pixel past
1932 the image is downscaled back to the same 4,761-patch ceiling, so **1932 x 1932 is
the largest un-downscaled square** - measured at the adjacent boundary, not inferred.

The ceiling is not a pure patch budget: a 1456x2576 canvas is also 4,784 patches and
came back at 0.623 of nominal, implying a downscale to roughly 1149x2033. A separate
long-edge limit applies. Moderate rectangles below that limit are unmeasured, so
rectangular geometry is not refuted in general - only that one extreme case. All
geometry probes ran on the Sonnet CLI path; other endpoints are not assumed identical.

---

## 2. Capacity

At 1932x1932 (4,761 image tokens):

| cell | px/char | cols x rows | chars in one image | chars/token |
|---|---|---|---|---|
| 5x10 | 50 | 386 x 193 | **74,498** | 15.7 |
| 6x13 | 78 | 322 x 148 | 47,656 | 10.1 |
| 8x16 | 128 | 241 x 120 | 28,920 | 6.1 |

These are *gross* capacities - characters physically rendered. What comes back correctly
is section 3.

**Against text tokenisation** (measured the same way, by billing delta):

| payload | chars per text token |
|---|---|
| typical English | ~4.0 |
| this template prose (codes, numbers) | 2.17 |
| random 4-char groups | **1.14** |

High-entropy data tokenises worst as text, so it has the most to gain from an image -
the opposite of the intuition, and the opposite of where fidelity holds up.

**Fill dominates font choice.** Reflowed text reaches 98% cell occupancy and 15.3
chars/token; the same text keeping its original line breaks drops to 41% and 6.7. That
swing is larger than any glyph-size decision. Reflow is safe for prose and destructive
for code, tables, hex dumps and diffs, so `pack_text.py --keep-lines` trades density for
structure where structure carries meaning.

---

## 3. Exact recovery

This is the section that matters, and the one where the exploratory numbers were most
misleading. Similarity scores of 0.95-0.98 correspond to an exact-match rate of zero:
a CER of 0.018 on a 56-character record is one wrong character, which is a fine
similarity score and a useless hash.""")

# ---- confirmatory table
if confirm:
    man = json.load(open(sorted(glob.glob(os.path.join(ROOT, 'results_confirm-v1_*.json')))[0]))['manifest']
    ALPHA = {'prose': 'closed-vocabulary English', 'alnum_easy': 'A-Z2-9, no I/O/0/1',
             'hex': '0-9a-f', 'b64': 'A-Za-z0-9+/ (I, l, O, 0 present)'}
    BASE = {'clR5x10', '6x13', '8x16'}
    w("", "### Whole records (56 chars, `confirm.py`)", "",
      "Base matrix: 5x10, 6x13, 8x16 x 2 reps. Conditions are heterogeneous, so per-payload",
      "rows are shown separately rather than pooled into one rate.", "",
      "| payload | alphabet | decode exact | mean CER | bind exact | bind hit a real row |",
      "|---|---|---|---|---|---|")
    for pl in ['prose', 'alnum_easy', 'hex', 'b64']:
        d = [r for r in confirm if r['payload'] == pl and r['probe'] == 'decode' and r['font'] in BASE]
        b = [r for r in confirm if r['payload'] == pl and r['probe'] == 'bind' and r['font'] in BASE]
        if not d: continue
        fr = sum(1 for r in b if r['displacement'] is not None)
        w(f"| {pl} | {ALPHA[pl]} | **{sum(r['exact'] for r in d)}/{len(d)}** | "
          f"{st.mean(r['cer'] for r in d):.3f} | {sum(r['exact'] for r in b)}/{len(b)} | {fr}/{len(b)} |")
    hi_base = [r for r in confirm if r['probe'] == 'decode' and r['payload'] != 'prose' and r['font'] in BASE]
    hi_ext  = [r for r in confirm if r['probe'] == 'decode' and r['payload'] != 'prose' and r['font'] not in BASE]
    w("", f"High-entropy decode across the base matrix: **{sum(r['exact'] for r in hi_base)}/{len(hi_base)}** exact.",
      f"A larger-cell extension (9x18, 10x20, 12x24, hex and base64 only) adds",
      f"{sum(r['exact'] for r in hi_ext)}/{len(hi_ext)} - the single success was hex at 12x24.",
      f"Combined that is {sum(r['exact'] for r in hi_base+hi_ext)}/{len(hi_base)+len(hi_ext)}, but the",
      "conditions differ and the pooled ratio should not be read as one probability. Per condition",
      "n=2, and 0/2 supports only a one-sided 95% upper bound of 0.776.")
    w("", "**Bigger glyphs do not fix it.** Across 5x10, 6x13, 8x16, 9x18, 10x20 and 12x24,",
      "mean hex CER moves 0.071 -> 0.018 while pixels per character rise 5.8x. Exact match",
      "stays at 0/2 in every cell but one (12x24 hex, 1/2). The residual is a composite -", "",
      "    D_total = D_visual + D_sequence + D_boundary + D_binding + D_abstention", "",
      "with evidence for each: base64 `f`/`F` confusions (visual), long records failing where",
      "short fields do not (sequence), one 51-character answer with 50 correct characters and",
      "only the last wrong (boundary), returned text matching no row (binding), and outright",
      "UNREADABLE responses (abstention). Optical resolution still matters - the f/F case",
      "proves it - so 'the residual is not optical' would be too strong.")

# ---- screening
if span:
    w("", "### Short spans, n=20 per condition (`span_probe.py`)", "",
      "Lowercase hex and Crockford base32 at 6x13, payload rendered in 4-character groups,",
      "asked for 'the next N characters' from an anchored start:", "",
      "| alphabet | symbols | span 8 | span 16 | span 32 | span 51 |", "|---|---|---|---|---|---|")
    for f, name, card in [('results_span_6x13_hex_opus.json', 'hex', 16),
                          ('results_span_6x13_crock32_opus.json', 'crock32', 32)]:
        d = span.get(f)
        if not d: continue
        cells = []
        for n in (8, 16, 32, 51):
            rs = [r for r in d if r['span'] == n]
            cells.append(f"{sum(r['exact'] for r in rs)}/{len(rs)}" if rs else '-')
        w(f"| {name} | {card} | " + ' | '.join(cells) + " |")
    w("", "One-sided 95% Clopper-Pearson lower bounds for hex: span 8 -> 0.717, span 16 -> 0.599,",
      "span 32 -> 0.544. An earlier 3-rep run of the same condition returned 3/3 and was",
      "reported as 'byte-exact through 32 characters'. At n=20 it is 15/20. That was small-sample",
      "luck in the favourable direction - the reason every rate here carries a bound.")

# ---- delimited
if delim:
    def rate(b, e, field='value'):
        rs = [r for r in delim if r['bits'] == b and r['enc'] == e]
        return sum(r[field] for r in rs), len(rs)
    def paired_ci(b):
        """Boundary-safe matched-pair risk-difference interval."""
        h = {r['rep']: r['value'] for r in delim if r['bits'] == b and r['enc'] == 'hex'}
        z = {r['rep']: r['value'] for r in delim if r['bits'] == b and r['enc'] == 'b32'}
        D = [int(h[k]) - int(z[k]) for k in h if k in z]
        lo, hi = paired_difference_interval(D)
        return sum(D) / len(D), lo, hi
    def concordance(b):
        h = {r['rep']: r['value'] for r in delim if r['bits'] == b and r['enc'] == 'hex'}
        z = {r['rep']: r['value'] for r in delim if r['bits'] == b and r['enc'] == 'b32'}
        pr = [(h[k], z[k]) for k in h if k in z]
        n = len(pr)
        obs = sum(1 for a, c in pr if a == c) / n
        ph = sum(a for a, c in pr) / n; pz = sum(c for a, c in pr) / n
        indep = ph * pz + (1 - ph) * (1 - pz)
        a11 = sum(1 for x, y in pr if x and y); a10 = sum(1 for x, y in pr if x and not y)
        a01 = sum(1 for x, y in pr if not x and y); a00 = sum(1 for x, y in pr if not x and not y)
        den = math.sqrt((a11+a10)*(a01+a00)*(a11+a01)*(a10+a00))
        phi = (a11*a00 - a10*a01) / den if den else float('nan')
        return obs, indep, phi
    def mcnemar(b):
        h = {r['rep']: r['value'] for r in delim if r['bits'] == b and r['enc'] == 'hex'}
        z = {r['rep']: r['value'] for r in delim if r['bits'] == b and r['enc'] == 'b32'}
        pairs = [(h[k], z[k]) for k in h if k in z]
        bb = sum(1 for a, c in pairs if a and not c)
        cc = sum(1 for a, c in pairs if c and not a)
        try:
            from scipy.stats import binomtest
            pv = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
        except Exception:
            pv = float('nan')
        return bb, cc, pv, len(pairs)
    paired = not os.path.exists(os.path.join(ROOT, 'results_delim_UNPAIRED_6x13_opus.json')) or \
             os.path.exists(os.path.join(ROOT, 'results_delim_6x13_opus.json'))
    w("", "### Delimited, equal-information fields, n=20 (`delim_probe.py`)", "",
      "The screening comparison above is confounded: 32 hex characters carry 128 bits, 32 base32",
      "characters carry 160, and scoring was literal. This one holds information constant, puts",
      "the field in visible brackets so nothing has to be counted, and scores decoded **value**",
      "equality (case-folding, separator removal and Crockford ambiguity mappings are part of the",
      "decoder, so they are not a grading loophole - but they must then be part of the protocol",
      "specification).", "",
      "**Decoder audit.** Of 34 value-correct base32 responses:", "",
      "| form | count | status |", "|---|---|---|",
      "| canonical literal output | 26 | exact string match |",
      "| required the permitted `O` -> `0` alias | 8 | valid *tolerant input*, not canonical output |",
      "| required a forbidden non-canonical form, range truncation or excess-bit masking | **0** | none |", "",
      "Those three rows are different things, and 'no non-canonical acceptances' would blur them.",
      "What is established is that no acceptance depended on discarding excess bits - the second",
      "row is the protocol working as specified, not the scorer over-accepting.", "",
      "Every alias rescue was the same confusion: `O` read for the printed digit `0`, 8 of 8. No",
      "case-only, no separator-only, no `I`/`L`. Since the Crockford alphabet excludes `O` from",
      "ground truth entirely, the model emits a symbol that cannot occur there.", "",
      "**Protocol decoder, now fixed.** The low-level `dec_b32` primitive has no bit-width and",
      "therefore returns the full integer. The width-aware `decode_value` path used by the scorer",
      "now enforces exact symbol count, `0 <= v < 2^B`, and canonical re-encoding before the tag",
      "or expected-value comparison. Production must use that width-aware path, not `dec_b32` alone.", "",
      "| bits | alphabet | chars | literal | value | rate | 95% LB |", "|---|---|---|---|---|---|---|")
    for b in (64, 128):
        for e in ('hex', 'b32'):
            rs = [r for r in delim if r['bits'] == b and r['enc'] == e]
            if not rs: continue
            v = sum(r['value'] for r in rs)
            star = '**' if v == len(rs) else ''
            w(f"| {b} | {e} | {rs[0]['chars']} | {sum(r['literal'] for r in rs)}/{len(rs)} | "
              f"{star}{v}/{len(rs)}{star} | {v/len(rs):.2f} | {lb(v, len(rs)):.3f} |")
    w("", "**No encoding advantage was detected - which is not the same as equivalence.** The",
      "difference in value-correct rate, with 95% intervals:", "")
    w("| bits | hex - base32 | 95% CI (matched-pair) | McNemar (discordant pairs) |",
      "|---|---|---|---|")
    for b in (64, 128):
        d0, lo, hi = paired_ci(b)
        bb, cc, pv, npair = mcnemar(b)
        w(f"| {b} | {d0:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {bb} vs {cc}, p = {pv:.3f} |")
    w("", "Intervals are matched-pair, not independent-sample: the same bitstrings are encoded",
      "both ways, so inference is based on the two discordant cells. With 1 and 5 discordant",
      "pairs, McNemar has almost no power - `p = 1.000` means the discordant evidence is too",
      "sparse to separate the encodings, not that they are equal. The displayed interval uses",
      "Bonferroni-simultaneous exact bounds on the two discordant cells, so it remains",
      "boundary-safe even when one discordant direction has count zero.")
    w("", "Those intervals are wide enough to contain a materially better hex *and* a modestly",
      "better base32. Establishing equivalence would need a pre-declared margin (say d = 0.10)",
      "and an interval falling entirely inside [-d, +d]; n=20 cannot do that. The earlier wording",
      "here, 'statistically indistinguishable', overstated a null result and is corrected.", "",
      "**Instrument defect, now fixed.** The first version of this probe put the encoding name in",
      "the RNG seed, so hex and base32 encoded *different* random values - the comparison was",
      "unpaired, and an earlier version of this README wrongly claimed the same values were used.",
      "The seed no longer includes the encoding, so both alphabets encode identical bitstrings and",
      "McNemar applies. The unpaired run is kept as `results_delim_UNPAIRED_6x13_opus.json`.", "",
      "**Between-run swings are large, but confounded.** The 128-bit hex condition scored 11/20 in",
      "the unpaired run and 16/20 in the paired one. Because the seed fix also changed the random",
      "values, that 0.25 swing mixes item-set variation with model and provider stochasticity - it",
      "is *between-dataset* variation under the same nominal condition, not an estimate of",
      "repeated-call variance. Isolating the latter needs the identical 20 frozen images rerun",
      "several times. The 64-bit hex cell scored 20/20 in both, i.e. 40/40 across two distinct item",
      "sets (unadjusted one-sided 95% LB 0.928). That is **replication across item sets within the",
      "same pipeline**, not independent confirmation: both runs share model family, prompt,",
      "renderer, endpoint, field layout, grader and selection history. It justifies freezing the",
      "format as the prototype candidate, nothing stronger.", "",
      "Pairing shows elevated concordance, but less than it first appears. At 128 bits the two",
      "encodings agree on " + f"{concordance(128)[0]:.2f}" + " of pairs against " +
      f"{concordance(128)[1]:.2f}" + " expected under independence - an excess of only " +
      f"{concordance(128)[0]-concordance(128)[1]:.2f}" + ", phi = " + f"{concordance(128)[2]:.2f}" + ".",
      "That is suggestive of a shared per-item or per-call difficulty component, **not** an",
      "identified value effect: shared outcomes could equally come from target position,",
      "surrounding page composition, or call-level variation. At 64 bits hex has no failures, so",
      "no association can be estimated at all. Separating them needs each bitstring repeated",
      "across positions, pages and calls in both encodings, with value / position / call as",
      "random effects.", "",
      "### Information-normalised throughput", "",
      "Raw success rate is the wrong objective: base32 carries the same information in fewer",
      "characters, so the comparison should be successfully decoded **bits** per image token,",
      "`rho_chars x (B/L) x P_value` at `rho_chars = 10.1`:", "",
      "| payload | chars | P(value) | successful bits/image-token |", "|---|---|---|---|")
    for b in (64, 128):
        for e in ('hex', 'b32'):
            k, n = rate(b, e)
            rs = [r for r in delim if r['bits'] == b and r['enc'] == e]
            if not rs: continue
            L = rs[0]['chars']
            w(f"| {b}-bit {e} | {L} | {k/n:.2f} | **{10.1 * (b / L) * (k / n):.1f}** |")
    w("", "On point estimates base32's shorter representation offsets its lower first-pass rate, so",
      "the alphabet question is open rather than settled in hex's favour.", "",
      "**This is anchored *payload* goodput, not page-level goodput.** It assumes the packer",
      "actually reclaims the three saved cells. If handles sit in a fixed 16-character slot for",
      "visual alignment then `L_physical = 16` for both encodings and base32's advantage mostly",
      "disappears. It also excludes delimiters, separators, page and block labels, tag bits,",
      "unused cells, retries and binding failure. The decisive comparison must be computed from",
      "complete rendered protocol pages:", "",
      "    G_page = sum_i B_i x 1[record i correctly resolved] / V(complete page)", "",
      "### The best observed field is a candidate, not a guarantee", "",
      "A visibly delimited 16-character hex field (64 bits) at 6x13 returned 20/20 value-correct.",
      "Pointwise that is a one-sided 95% lower bound of 0.861 - but it is the best of four",
      "delimited conditions, selected after a long adaptive exploration of cells, layouts,",
      "alphabets and span lengths. A four-cell Bonferroni adjustment alone would drop the bound to",
      "about 0.803, and the wider selection history is not repairable by any adjustment. The",
      "defensible statement is **best observed candidate**, pending one fresh held-out run of the",
      "now-frozen design. On successfully decoded bits per image token the 13-character base32",
      "field actually leads it (47.2 against 40.4), so 'best' depends on which objective is being",
      "optimised - exact-field rate or information throughput.", "",
      "It is also a *raw* 64-bit value, not a checksummed handle. A handle must spend those bits",
      "as a budget across record identity, context binding and error detection:", "",
      "For a t-bit keyed tag, `P_FA <= M / 2^t` where **M is the number of *distinct* wrong",
      "candidates validated** - not the number of model calls, since repeatedly producing the same",
      "bad candidate is not an independent trial. The bound assumes a truncated secure keyed MAC",
      "or PRF, a key untouchable by untrusted content, canonicalisation before verification, and a",
      "tag binding all relevant context.", "",
      "| tag width | false accepts over 1e6 distinct bad candidates | remaining bits for identity |",
      "|---|---|---|",
      "| 32-bit | ~2.3e-4 | 32 |", "| 40-bit | ~9.1e-7 | 24 |", "| 48-bit | ~3.6e-9 | 16 |", "",
      "A hierarchical protocol cuts the identity requirement: with page and block already",
      "established, a 16-bit *local* record index plus a 48-bit keyed tag still fits the measured",
      "64-bit span. The tag should cover archive, page, block, record index and a record",
      "fingerprint, so a handle lifted from the wrong page fails validation **when the expected",
      "context is independently known**. It cannot catch a correctly-read handle for the wrong",
      "record inside an accepted context - only reconciliation after fetch does that.", "",
      "A structured codeword also has a different symbol distribution from a uniform random value,",
      "so the 20/20 result does not transfer to it untested.", "",
      "### Grouping and delimiting interact with length", "",
      "| field | grouped payload, counted span | contiguous payload, delimited |",
      "|---|---|---|", "| 16 chars | 16/20 | **20/20** |", "| 32 chars | 15/20 | **11/20** |", "",
      "Removing the counting burden helped at 16 characters and hurt at 32, so 15/20 was not a",
      "'floor' for delimited designs. Two variables changed together (grouping vs contiguity, and",
      "counted vs delimited), so no separate effect is identified - an earlier claim here that",
      "visible grouping stabilises the per-symbol rate is unsupported and withdrawn. The opposite",
      "directions at 16 and 32 characters also rule out one constant per-character success",
      "probability across layouts. The minimum causal experiment is a paired 2x2:", "",
      "    {grouped, contiguous} x {counted, delimited},  at 16 and 32 chars,",
      "    identical underlying values and identical image positions", "",
      "which separates grouping benefit, delimiter benefit, counting cost, and their interaction",
      "with length.", "")

# ─────────────────────────────────────────────────────────────── error channel
if span.get('results_span_6x13_hex_opus.json'):
    h = span['results_span_6x13_hex_opus.json']
    obs = {n: sum(r['exact'] for r in h if r['span'] == n) for n in (8, 16, 32)}
    try:
        from scipy.optimize import minimize_scalar
        def nll(q):
            t = 0
            for n, k in obs.items():
                p = min(max(q ** n, 1e-12), 1 - 1e-12)
                t -= k * math.log(p) + (20 - k) * math.log(1 - p)
            return t
        q = minimize_scalar(nll, bounds=(0.90, 0.9999), method='bounded').x
    except ImportError:
        q = 0.989
    w("", "---", "", "## 4. The channel is sparse symbol substitution", "",
      f"Fitting `P_exact(n) = q^n` jointly across spans 8/16/32 for grouped lowercase hex at 6x13:", "",
      f"    q = {q:.4f}      per-symbol error {100*(1-q):.2f}%", "",
      "| span | observed | q^n predicted |", "|---|---|---|")
    for n in (8, 16, 32, 51):
        k = obs.get(n)
        w(f"| {n} | {(f'{k}/20' if k is not None else '**0/20**')} | {q**n:.3f} |")
    p51 = (1 - q ** 51) ** 20
    w("", f"Spans 8-32 fit one per-symbol rate closely. Span 51 does not: under the fitted model",
      f"`P(0/20) = {p51:.1e}`. The 51-character task is a **separate failure regime** - counting,",
      "boundary tracking, output-span control or abstention - not more of the same optical noise.",
      "(An earlier reading of the 3-rep data claimed the spans refuted a per-character model; with",
      "n=20 the model fits, and that claim is withdrawn.)", "",
      "Failures are also sparse: **1.0 character edits per failed record** at spans 16 and 32, 2.0",
      "at span 8. Nearly every failure is a single wrong symbol.", "",
      "**So the engineering response is detection and correction, not pixels.** Larger cells did",
      "not move CER; a single-symbol-correcting code over a short field would recover almost every",
      "observed failure. The metric that then matters is not `P(literal mismatch)` but", "",
      "    P(a corrupted handle resolves to a DIFFERENT valid record)", "",
      "driven to zero by checksum plus unique-match rejection - never by trusting the transcription.")

# ─────────────────────────────────────────────────────────────── binding
w("", "---", "", "## 5. Binding is the dominant failure", "",
  "Decoding correctly says nothing about *provenance*. The bind probe asked for a named row",
  "and scored which row actually came back:", "",
  "| payload | bind exact | returned text matched no row at all |", "|---|---|---|",
  "| prose | 3/6 | 2/6 |", "| alnum | 0/6 | 5/6 |", "| hex | 0/6 | 3/6 |", "| b64 | 0/6 | 5/6 |", "",
  "In the exploratory ladder the same effect appeared as legibility 1.00 with lookup 0%: the",
  "model transcribed an adjacent row perfectly. That is a referential error, not an OCR error,",
  "and it is more dangerous because the returned string is well-formed.", "",
  "Two limits on reading this. 0/6 per condition gives a one-sided 95% upper bound of 0.393,",
  "so 'near zero' overstates it; and pooling heterogeneous payloads to n=18 is invalid for",
  "estimating one probability. What is established is that **dense row-number addressing is a",
  "bad protocol**, not that binding is an intrinsic optical limit.", "",
  "Page shape does not rescue it. At a fixed cell, 74-row, 112-row and 149-row pages all gave",
  "legibility 0.93-0.97 and lookup accuracy 0-25%. Fewer rows did not help.", "",
  "Hierarchical addressing - page ID, block ID every 8-16 records, short record handle, with a",
  "textual manifest outside the bitmap - is **untested** and is the largest open item here.", "",
  "### A checksum does not fix this", "",
  "Two failure classes look identical from outside and only one is detectable:", "",
  "| class | what happens | detectable by checksum |",
  "|---|---|---|",
  "| lexical corruption | right record selected, handle misread: `H_i -> H_hat != H_i` | **yes** |",
  "| provenance corruption | wrong record selected, its *valid* handle read correctly: `R_j -> H_j` | **no** |", "",
  "The second passes validation because `H_j` is a well-formed codeword for a real record. So a",
  "validation tag solves transcription failure and does nothing for binding failure, which is",
  "the one the bind probe actually found. Any production path needs semantic or structural",
  "reconciliation after the canonical fetch - confirming the retrieved record is the object that",
  "was asked for - not just a passing checksum.", "",
  "---", "", "## 6. Density taxonomy", "",
  "| quantity | 6x13 value | meaning |", "|---|---|---|",
  "| `rho_gross` | 10.1 | characters physically rendered per image token |",
  "| `rho_symbol` | ~10.0 | expected *correct* symbols per image token (CER 0.008) |",
  "| `rho_anchored` | ~7.6 | all-or-nothing record yield at the 32-char success rate |",
  "| `rho_bits` | ~40-42 | successfully decoded *bits* per image token, short fields |",
  "| `rho_operational` | **unknown** | includes binding, checksum rejection, retries, fetch |", "",
  "The end-to-end quantity is bits of *canonical* information resolved per visual token:", "",
  "    rho_resolved = rho_bits x P(correct record bound)",
  "                            x P(validation accepts only when it should)",
  "                            x P(canonical fetch returns the right record)", "",
  "Only the first factor is measured. `P(false accept)` - a corrupted code validating as a",
  "different real record - is not measured at all, and is the number that decides whether the",
  "channel is safe rather than merely dense.", "",
  "`rho_anchored` assumes one correctly anchored record, zero utility for partial reads, and no",
  "cost for retries or canonical fetch. It is a throughput proxy, not a channel capacity. An",
  "earlier version of this file called it `rho_operational`; the span probe supplied the anchor,",
  "so it never measured `P(correct anchor)` at all.")

# ─────────────────────────────────────────────────────────────── exploratory
w("", "---", "", "## 7. What limits density (exploratory)", "",
  "> Everything in this section is **exploratory**. The prompt, tool policy, grader, payloads,",
  "> line lengths and candidate fonts were changed during the run in response to results, and",
  "> the reporting threshold was chosen after seeing scores. Treat it as hypothesis generation.",
  "> Sections 1-5 are the measured claims.", "",
  "**Legibility is not one variable.** Separating glyph resolution from row addressing, and",
  "redundant text from unguessable text, splits the apparent 'floor' into three different limits:", "",
  "* **Row pitch** matters more than glyph area. 6x9 (54 px/char) reads where clR8x8 (64 px/char,",
  "  larger glyphs) fails - everything at 8px pitch or below failed, everything at 9px and above",
  "  read. But pitch alone is not sufficient: 4px-wide glyphs failed at pitch 9 and 10 and only",
  "  recovered at pitch 12, so width and pitch interact.",
  "* **Payload redundancy** is worth more than any font choice. At 6x10 - a cell that reads",
  "  template prose at ~0.95 - the redundancy credit against unguessable payload reached +0.94.",
  "  A closed-template corpus lets a model pattern-match rather than resolve glyphs.",
  "* **Line length** confounds legibility with output stamina. The same cell scored 0.59 on",
  "  224-character lines and 0.95 on 56-character lines.", "",
  "The ladders themselves (bitmap 4x6 to 12x24, TrueType 5-32px, proportional prose) live in",
  "`images/A_*`, `B_*`, `C_*` with per-image density in `manifest.csv`.")

# ─────────────────────────────────────────────────────────────── method
if confirm:
    w("", "---", "", "## 8. Method", "",
      f"The confirmatory harness freezes everything before running: prompts (sha",
      f"{man['prompt_decode_sha']} / {man['prompt_bind_sha']}), harness (sha {man['harness_sha']}),",
      f"grader, tool policy, line length ({man['line_chars']} chars), and a fresh seed namespace.",
      f"Raw responses are retained. CLI {man['cli_version']}.", "",
      "* **Tools are locked to `Read`.** Without that, a model will shell out to Python to crop and",
      "  zoom the image, which measures tooling rather than vision. That happened, and is why",
      "  `Bash,Write,Edit,Glob,Grep,Task,WebFetch,WebSearch` are explicitly disallowed.",
      "* **Abstention is distinguishable from error.** Prompts ask for `UNREADABLE` rather than a",
      "  guess, and the corpus is random enough that a guess is always wrong.",
      "* **Grading separates decoding from addressing.** For any wrong answer the grader searches",
      "  every ground-truth line for what the model actually returned, so reading the right",
      "  characters off the wrong row is scored as a binding error, not a legibility one.",
      "* **Value vs literal equality.** Hex and base32 tolerate case-folding and regrouping without",
      "  changing the decoded value, so both are reported. For base64 case is semantic and no such",
      "  normalisation is permitted.",
      "* **n is small.** Two repetitions per cell/payload in the confirmatory matrix, 20 in the",
      "  screening and delimited stages. 59/59 clean runs would be needed to put a one-sided 95%",
      "  bound above 0.95, and 299/299 above 0.99. Nothing here reaches that.")

# ─────────────────────────────────────────────────────────────── confounds
w("", "---", "", "## 9. Confounds found, including in these instruments", "",
  "Most of the useful findings here came from discovering that a measurement was measuring",
  "something else.", "",
  "| confound | effect | fix |", "|---|---|---|",
  "| Rater shared context with the corpus generator | reading becomes recall with a visual hint | fresh `claude -p` process per run |",
  "| Closed-template corpus | inflates legibility up to +0.94 vs unguessable payload | series K/K2 high-entropy payloads |",
  "| Confusable-free alphabet (no I/O/0/1) | optimistic for real hashes and hex | hex / base64 / base32 added |",
  "| 224-character lines | measures output stamina, not reading | line length held at 56 |",
  "| 'transcribe the next N characters' | measures counting too | delimited fields |",
  "| Equal character count across alphabets | 160 bits vs 128 - not a fair comparison | equal-information encoding |",
  "| Literal string scoring | penalises value-preserving case/grouping | value equality reported alongside |",
  "| Model given non-Read tools | crops and zooms instead of reading | tools disallowed |",
  "| Adaptive test selection, post-hoc threshold | winner's curse | frozen confirmatory pass |", "",
  "The original audit found six harness bugs that corrupted results before being caught. From the Claude arm: Pillow",
  "mis-decodes X11 PCF fonts declaring `first_col > 0` (8x16 and 12x24 rendered one glyph",
  "shifted - patched in `gen_lib.py`); `edge_probe.py` overwrote its results file each run; and",
  "stored answers were truncated to 120 characters before comparison against full-length ground",
  "truth. From the GPT-5.6 arm:", "",
  "| bug | symptom | fix |", "|---|---|---|",
  "| Prompt caching hid image tokens | a repeat call puts the image in `cached_input_tokens`, so `input + cache_creation` collapsed to ~0 and the first Opus geometry probe returned garbage | render **unique content for every measurement call** so the image can never be a cache hit |",
  "| `codex exec -i` is variadic | a positional prompt placed after `-i` is swallowed as another image file; the call failed with 'No prompt provided via stdin' | pass the prompt on **stdin**, which also avoids argv limits on long prompts |",
  "| `tokens()` omitted the 1.2 multiplier | GPT-5.6 densities in the answer key were inflated by exactly 1.2x, putting some images *above* their theoretical ceiling | recomputed every token field from geometry; **156 of 161 entries were wrong** |", "",
  "The third is the most instructive: it was caught only because a sanity check asserted that no",
  "image may exceed `1024/1.2/(cw*ch)` chars per token. Without that assertion the inflated",
  "figures would have looked plausible and propagated into every Part II conclusion.", "",
  "A later reproducibility audit found additional defects. It traced each one through the stored",
  "artifacts before deciding whether a reported count had changed:", "",
  "| later defect | impact on stored claims | resolution |", "|---|---|---|",
  "| The documented build stopped after 104 of 161 cases | the committed corpus could not be reproduced by the stated command, although its image bytes were sound | one build path now emits all 28- and 32-grid series; catalog hashes are validated by an isolated rebuild |",
  "| `confirm.py --effort` recorded the requested effort but workers always used `low` | no stored count changed because every existing confirm file requested `low` | forward the argument and cover it with a regression test |",
  "| Four extension manifests advertised the base matrix | result rows were sound but their cells and payload metadata were wrong | correct the legacy manifests from their actual rows |",
  "| `UNREADABLE` on an absent query was graded `N` | this could inflate safe abstention; every stored `N` was audited as a literal `NO_MATCH`, so published counts did not change | grade it `P0` and test the absent-item branch |",
  "| The stored image and text Stage-1 runs used different item namespaces | the apparent same-item comparison and its Fisher tests were invalid; the query sets have zero overlap | withdraw those claims and require paired items in the held-out campaign |",
  "| Hex decoders discarded arbitrary non-hex characters | malformed answers could be rescued; no stored classification depended on that rescue | accept only documented separators, exact decoded length, and canonical value |",
  "| Raw responses and provider identity were recorded inconsistently | exact legacy model revisions cannot be recovered from mutable aliases | versioned result envelopes now retain hashes, environment, full stdout/stderr, usage, and immutable model identity; aliases are rejected by default |",
  "| A stale 1.15 MP Claude flag survived in the catalog | it contradicted the later geometry result | regenerate it from the measured provider geometry |", "",
  "---", "", "## 10. What was retracted", "",
  "| claim | status | replaced by |", "|---|---|---|",
  "| 1568x1568 is the maximum canvas | **retracted** | 1932x1932, measured at the adjacent boundary |",
  "| Canvas must be a multiple of 224 | **narrowed** | `lcm(28, cw)` / `lcm(28, ch)` for Claude alone |",
  "| Claude downscales above ~1.15 MP | **retracted** | no downscaling to 3.63 MP on this path |",
  "| Row pitch >= 9 is the rule | **narrowed** | necessary, not sufficient - 4px glyphs fail at pitch 9-10 |",
  "| 0.95-0.98 similarity means near-exact | **retracted** | exact match 0/18 at those scores |",
  "| P(exact) = 0 for high-entropy payload | **narrowed** | no tested config demonstrated it; 0/18 gives UB 0.153 |",
  "| Bigger glyphs do not fix exactness | **narrowed** | composite failure; optical resolution still matters |",
  "| 9/9 exact through 32 characters | **retracted** | 15/20 at n=20 - small-sample luck |",
  "| The span data refutes a per-character model | **retracted** | q = 0.989 fits spans 8-32 |",
  "| Alphabet size dominates confusion-resistant design | **retracted** | indistinguishable at equal information |",
  "| 15/20 is a floor for delimited fields | **withdrawn** | delimited scored worse at 32 chars |",
  "| rho_operational ~= 10 chars/token | **renamed** | `rho_anchored`; operational still unknown |",
  "| 8x16 is safe for anything | **retracted** | tested ASCII only, and 0/6 exact on whole records |",
  "| *Part II* | | |",
  "| A 32x32 patch costs 1 token | **retracted** | GPT-5.6 bills 1.2 tokens/patch; 853 px/token, not 1024 |",
  "| Claude's 1932px ceiling is a general limit | **retracted** | GPT-5.6 caps at 1600x1600 / 2,500 patches |",
  "| The span-51 cliff is a property of vision models | **retracted** | Claude-specific; all three 5.6 models read it |",
  "| Enlarging glyphs cannot fix exactness | **narrowed to Claude** | terra reaches 8/12 on large cells |",
  "| 'GPT-5.6 reads dense text better' | **retracted as a family claim** | only sol; luna and terra are significantly worse |",
  "| Sol's abstention failure is family-wide | **retracted** | sol only; terra and luna abstain correctly |",
  "| Better legibility implies better record resolution | **retracted** | sol leads the tested reading matrix; record resolution remains unresolved because Claude has only a debug preflight and sol only Stage 1 |")

# ─────────────────────────────────────────────────────────────── files
w("", "---", "", "## 11. Files", "",
  "| script | what it does |", "|---|---|",
  "| `providers.py` | one call interface over `claude -p` and `codex exec`; usage extraction |",
  "| `geometry.py` | provider-agnostic px/token and downscale-ceiling probe |",
  "| `gen_lib.py` | fonts, token maths, `PROVIDER_GEOMETRY`, corpus generators, PCF fix |",
  "| `generate.py` | the 92-image exploratory corpus (series A-G) |",
  "| `pack.py` | zero-waste canvas solver; series H-K2 |",
  "| `pack_text.py` | packs arbitrary text into the fewest image tokens |",
  "| `run_eval.py` | ladder eval, addressing-aware grader |",
  "| `edge_probe.py` | legibility isolated from row addressing |",
  "| `confirm.py` | frozen confirmatory pass, exact match + binding displacement |",
  "| `span_probe.py` | exact match vs requested span length |",
  "| `delim_probe.py` | delimited equal-information fields, value equality |",
  "| `ebind1.py` | full protocol test: keyed codewords, canonical fetch, gold scoring |",
  "| `regrade.py` | audits stored answers without new API calls; writes only with `--write` and records provenance |",
  "| `compare.py` | cross-provider tables on paired, byte-identical stimuli |",
  "| `analyze_ebind.py`, `analyze_encoding.py`, `analyze_layout.py`, `analyze_verifier.py` | clustered, equivalence, factorial and verifier-safety analysis |",
  "| `analysis_ebind_stage1_sol.{json,md}` | corrected item/page-clustered Stage-1 analysis |",
  "| `campaign.py` | plans the 2,265-call held-out/Arm-B/verifier/layout/effort campaign; execution is explicit and resumable |",
  "| `verifier_probe.py` | fail-closed post-fetch reconciliation on correct and deliberately wrong-valid canonical records |",
  "| `validate_project.py` | catalog hashes, schemas and isolated byte-rebuild invariant |",
  "| `probe.py`, `report_eval.py`, `make_docs.py` | inspection and deterministic reporting |", "",
  "### Reproducible environment", "",
  "Python 3.11 dependencies are pinned in `requirements.lock` and declared in `pyproject.toml`.",
  "Corpus generation additionally needs DejaVu TrueType fonts and X11 bitmap fonts",
  "(`fonts-dejavu-core` and `xfonts-base` on Debian/Ubuntu); converted bitmap assets are vendored.",
  "Model execution needs the relevant `codex` or `claude` CLI and its normal configured",
  "credentials. Offline generation, validation, analysis and documentation make no external calls.", "",
  "```bash",
  "python3 -m venv .venv",
  ". .venv/bin/activate",
  "python -m pip install -r requirements.lock",
  "python -m pytest",
  "```", "",
  "```bash",
  "python3 generate.py && python3 pack.py      # build images",
  "python3 geometry.py --model gpt-5.6-sol --patch 32   # px/token + ceiling",
  "python3 confirm.py --model gpt-5.6-sol --effort low  # frozen pass, any provider",
  "python3 make_docs.py                        # rebuild this file",
  "python3 validate_project.py --rebuild       # offline integrity + byte identity",
  "python3 campaign.py --model <exact-id>      # plan only; no external calls",
  "```", "",
  "Content is seeded per image, so a rebuild reproduces byte-identical images and answers.", "",
  "---", "", "## 12. Open questions", "",
  "1. **Held-out cross-provider protocol comparison.** Hierarchical page -> block -> handle",
  "   addressing is implemented, but Claude and the non-sol models have only debug preflights.",
  "   Run the same frozen Stage-1 item set on the deployment candidates before ranking them.",
  "2. **The admissible canvas region.** One rectangle was tested and failed. The long-edge limit,",
  "   and whether moderate rectangles improve row count or cell divisibility, are unknown.",
  "3. **Delimited x grouped, factorially.** The two were changed together; their separate",
  "   contributions are unmeasured.",
  "4. **Short fields at larger cells.** The n=20 screen ran only at 6x13. 9x18 and 12x24 may",
  "   support longer exact fields; the whole-record CER result does not settle it.",
  "5. **Error-corrected handles and reconciliation.** Test a structured codeword - 32-bit record",
  "   index plus 32-bit keyed validation tag in the same 16-character span - and measure both",
  "   optical `P(false accept)` and a fail-closed semantic verifier on deliberately wrong-valid",
  "   fetched records. Both instruments are implemented; their held-out runs remain unexecuted.",
  "6. **Equivalence, not just non-significance.** Declare a margin (d = 0.10) and power the",
  "   hex/base32 comparison to fit the difference interval inside it. n=20 cannot.",
  "7. **The 2x2 layout experiment.** Grouping x delimiting at 16 and 32 characters, paired.",
  "8. **Other endpoints.** Geometry is measured on the Claude CLI and Codex attachment paths;",
  "   API and hosted endpoints remain unmeasured.", "",
  "---", "", "## 13. E-BIND-1: design", "",
  "> **Status: executed.** This section is the design and its rationale; measured results for",
  "> both providers are in section 23. Kept separate because the design decisions were frozen",
  "> before any data was collected, and that ordering is the point.", "",
  "Stop sweeping fonts, alphabets and isolated field accuracy. The remaining uncertainty is",
  "whether a semantic query can resolve the *correct canonical record*. Implemented in",
  "`ebind1.py`.", "",
  "### Pin one model - do not compose across them", "",
  "Every fidelity result in this file was measured on **Opus**; every *original* geometry result,",
  "including the 1932x1932 ceiling and the 784 px/token rate, was measured on **Sonnet**. That",
  "composition was unverified, and preprocessing, downscaling and effort behaviour can all differ",
  "by model. It has since been checked directly: the adjacent-boundary probe was repeated on both",
  "models and both cap at 4,761 patches with the same 1932 boundary (section 1), so the geometry",
  "transfers. The *fidelity* numbers remain Opus-only. E-BIND-1 still pins one exact model id,",
  "endpoint and CLI version for both halves, and mutable aliases (`opus`, `sonnet`) must not",
  "appear in the final provenance record.", "",
  "### Gold scorer is not a runtime verifier", "",
  "The evaluator knows which record generated each query, so it scores `R_fetched == R_gold`",
  "deterministically. That is the **primary** metric. A deployable system does not know",
  "`R_gold`, so any runtime semantic verifier is a separate component and must be measured as a",
  "second arm - reporting `P(verifier accepts | R_fetched != R_gold)`. Implementing the",
  "'semantic verifier' with the hidden answer key would make false acceptance impossible by",
  "construction and would describe nothing deployable.", "",
  "### Two outcome partitions, because abstention flips sign", "",
  "| answer-present | | no-answer | |", "|---|---|---|---|",
  "| `C` | correct gold record accepted | `N` | correct NO_MATCH |",
  "| `D` | wrong/corrupt candidate rejected | `W0` | **any record accepted** |",
  "| `W` | **wrong canonical record accepted** | `D0` | invalid candidate rejected |",
  "| `A` | abstained | `P0` | parse/protocol failure |",
  "| `P` | parse/protocol failure | | |", "",
  "The two load-bearing rates are `P(W | answer present)` and `P(W0 | no answer)`. Abstention is",
  "correct behaviour in one partition and failure in the other, so they are never pooled.", "",
  "### Two arms, because one layout cannot do both jobs", "",
  "* **Arm A - fixed slot.** Both encodings occupy the same physical 16-character slot, so record",
  "  positions, neighbours, line lengths and page geometry are identical. Base32's padding is",
  "  deliberately wasted. This isolates encoding accuracy.",
  "* **Arm B - native density.** Hex uses 16 cells, base32 13, pages packed independently. This",
  "  measures `G_page` and is an architectural comparison, not a paired glyph-level one.", "",
  "Asking one layout to do both would either shift the page (breaking pairing) or pad base32",
  "(erasing the density advantage it is supposed to demonstrate).", "",
  "### Structural IDs outside the bitmap - and the residual binding channel", "",
  "    ARCHIVE A1 - PAGE P07 - BLOCK B03: <image path>      <- ordinary exact text",
  "    [dense image payload]                                <- semantic content + record codewords", "",
  "Page and block identifiers are tiny; compressing them optically adds binding uncertainty for",
  "no density gain. But an exact *text* label can still be associated with the wrong adjacent",
  "image, so that channel is measured rather than assumed. The keyed tag binds page and block, so",
  "`P(tag pass | code read exactly)` reports label-image consistency for free: a code lifted from",
  "the wrong block fails validation.", "",
  "### Codeword", "",
  "8-bit local index + 56-bit keyed tag over",
  "`protocol || archive || revision || page || block || index || digest(record)`. Blocks hold",
  "8-16 records so 8 index bits suffice, and the archive revision binds a handle to a canonical",
  "state generation - a stale handle cannot silently resolve against changed state. Tag width is",
  "not the bottleneck: the dominant error is selecting another real record carrying its own valid",
  "tag, which no tag width addresses.", "",
  "### Sampling - repeated calls are not extra items", "",
  "100 unique answer-present queries + 20 no-answer, paired across encodings, x3 passes over the",
  "frozen set = 720 calls. That is **120 independent semantic items**, not 720. The passes",
  "estimate call variance conditional on those items; they do not multiply the item sample.",
  "Analysis is repeated-measures:", "",
  "    logit P(Y_ier = 1) = b0 + b_enc + u_item + v_page + w_pass", "",
  "with a hierarchical bootstrap over items and pages as an alternative. For generalisation to",
  "new items, `0/100` bounds the false-accept rate below only ~3%; the `0/299` -> ~1% and",
  "`0/2995` -> ~0.1% figures require that many **independently constructed adversarial binding",
  "opportunities**, not repeated calls over the same ones.", "",
  "### Required controls", "",
  "* **Wrong-valid-handle decoys**: semantically similar payload, different valid codeword, valid",
  "  tag, same block where possible. A checksum rejects lexical corruption and accepts these;",
  "  only post-fetch reconciliation rejects them. Without decoys a low false-accept count may",
  "  only mean the test never produced a plausible wrong candidate.",
  "* **Raw-text semantic ceiling**: identical queries, records, block structure, distractors and",
  "  no-answer cases, carried as ordinary text. Without it a 65% optical score cannot be split",
  "  into 95% semantic x 68% optical or 68% semantic x ~100% optical. One frozen pass suffices.",
  "* **Preflight (Stage 0)**: ~20 queries per encoding to validate range checking, canonical",
  "  re-encoding, tag construction, wrong-page/wrong-block rejection, same-block decoys reaching",
  "  the semantic scorer, the outcome partition, raw retention and deterministic regrading - then",
  "  freeze commit, renderer, prompts, model id and seed namespace, and regenerate a fresh",
  "  held-out corpus. Preflight results are debug-only and must not enter the estimate.", "",
  "### Stage 0 preflight result (debug-only, not an estimate)", "",
  "75 calls on Opus, Arm A, 6 blocks x 12 records, 20 answer-present + 5 no-answer queries:", "",
  "| carrier | C | D | W | N | W0 | P(correct block) |", "|---|---|---|---|---|---|---|",
  "| image, hex | **20/20** | 0 | **0** | 5/5 | **0** | 20/20 |",
  "| image, base32 | 17/20 | 3 | **0** | 5/5 | **0** | 17/20 |",
  "| text ceiling, hex | 20/20 | 0 | 0 | 5/5 | 0 | 20/20 |", "",
  "The semantic ceiling is 20/20, so the query set is unambiguous and the optical arms are not",
  "limited by semantic selection - hex matched the ceiling exactly. Compare this with row-number",
  "addressing (section 5), which scored 0/6 on the same kind of task: **exact text labels plus a",
  "short delimited keyed codeword is a working protocol where dense row lookup was not.**", "",
  "The three base32 losses were 2 tag failures and 1 length failure (a 12-symbol code - a dropped",
  "character), all clustered in one block. All three were *rejected*; none was accepted.", "",
  "**What this does not show.** No wrong canonical record was ever accepted, but the model never",
  "selected a wrong record, so the W path was never exercised by the model - exactly the",
  "circumstance in which a zero false-accept count means nothing. The path was therefore verified",
  "by direct injection instead:", "",
  "| injected response | outcome | validator |", "|---|---|---|",
  "| a different real record in the same block | **W** | `ok` - the tag passes, as it must |",
  "| the correct record | C | `ok` |",
  "| any record on a no-answer query | **W0** | `ok` |",
  "| a forged code | D | `tag_fail` |", "",
  "So the partition is exhaustive and the dangerous case is detectable; it simply did not arise",
  "at n=25 with 72 records. `0/20` bounds new-item false acceptance below only ~14%. Preflight",
  "results are debug-only and do not enter any estimate.", "",
  "### Reported metrics", "",
  "`P(correct image block)`, `P(correct record | correct block)`, `P(correct block | text",
  "ceiling)`, `P(canonical parse)`, `P(tag pass | correct record)`, `P(tag pass | corrupted",
  "code)`, the full `C/D/W/A/P` and `N/W0/D0/P0` partitions, `G_page`, expected retries, p50/p95",
  "latency and token cost. **`P(W)` is reported separately and never folded into a utility",
  "score.**", "",
  "---", "", "## 14. Assumption register", "",
  "| assumption | status | falsification probe |", "|---|---|---|",
  "| The 20 paired repetitions use fresh stimuli/calls | seeded values are unique by rep; provider-side call independence is not observable | retain item/call IDs and never count calls as extra items |",
  "| Pairing held target position and surrounding content constant | **audited in renderer tests**: same values, canvas and field start; only the assigned formatting factor changes | keep condition names out of the seed |",
  "| No acceptance depended on excess-bit masking or truncation | **audited: yes** (0 of 34) | canonical re-encode audit |",
  "| Tolerant normalisation is limited to protocol aliases | **audited: yes** (8 of 8 were `O`->`0`) | classify every literal-wrong/value-correct response |",
  "| The width-aware base32 path rejects over-range/non-canonical values | **audited: yes** | exact length, range and canonical re-encode are enforced before comparison/tag check |",
  "| Matched-pair intervals remain valid at sparse boundaries | **fixed** | simultaneous exact discordant-cell bounds replace the Wald interval |",
  "| 40/40 is independent confirmation | **too strong** - shared pipeline | treat as replication across item sets |",
  "| base32 saves cells at page level | **rendered in Arm B**; successful goodput remains unmeasured | execute paired Arm-B queries and include retries |",
  "| Exact-text page/block labels preserve selection difficulty | plausible | compare against in-image labels |",
  "| Semantic verification catches wrong valid records | instrumented, not yet measured | held-out fail-closed verifier arm with deliberately injected near-decoys |",
  "| Concordance reflects bitstring difficulty | suggestive only (phi = 0.29) | repeat each value across positions and calls |",
  "| The two runs measure stochastic variation | **false as stated** - values differed too | rerun identical frozen images |",
  "| base32 goodput advantage survives protocol overhead | plausible, unmeasured | include delimiters, manifests, validation, retries |",
  "| A structured 64-bit handle behaves like a uniform value | unknown; 32+32 instrument implemented | execute the held-out structured-codeword arm |",
  "| Hierarchical binding beats row addressing | promising in preflight; Stage 1 exists only for sol | held-out Stage 1 on each candidate |",
  "| A 32x32 patch costs one token | **false** - GPT-5.6 bills 1.2 tokens/patch | fit tokens against patch count (section 15) |",
  "| Claude's 1932px ceiling transfers to other providers | **false** - GPT-5.6 caps at 1600px | re-run `geometry.py` per provider |",
  "| The 5.6 line beats Claude on high-entropy exact decode | directional only, n=18, CIs overlap | span/delim stages at n=20 per condition |",
  "| The two providers are effort-matched | **false** - 5.6 pinned `low`, Claude uncontrolled | sweep effort on 5.6 |",
  "| Image delivery is equivalent across providers | **false** - attached vs path+Read tool | unavoidable; state it on every table |",
  "| The 5.6 line is uniform on this task | **false** - sol beats Claude on legibility, luna and terra are significantly worse | per-model paired tests (section 19) |",
  "| Long-span reading failure is intrinsic to vision models | **false** - Claude 0/20 at span 51, all three 5.6 models 12-16/20 | span sweep on both providers |",
  "| Bigger glyphs cannot fix exact recovery | **false for GPT-5.6** - terra 8/12 on large cells where opus manages 1/12 | confirm extension, both providers |",
  "| Uppercase base32 is workable | **false for the entire 5.6 line** - opus 17/20, sol 4/20, terra 2/20, luna 0/20 | four independent probes |",
  "| A model that reads better resolves records better | unresolved across providers - protocol rankings are preflight-only except sol | held-out Stage 1 |",
  "| Poor image results imply poor semantics | text control is 99%, but the stored Stage-1 text run used a different item namespace | rerun paired carriers |",
  "| Abstention failure is a 5.6 family trait | **false** - only sol; terra and luna abstain 5/5 | preflight across three models |",
  "| GPT-5.6 effort setting does not matter here | **unmeasured** - everything ran pinned at `low` | sweep low/medium/high on one condition |")

# ─────────────────────────────────────────────────────────────── write
readme = '\n'.join(S) + '\n'
write_output('README.md', readme)

cards = []
for k, v in E.items():
    cards.append(f"""<figure><a href="images/{k}"><img src="images/{k}" loading="lazy"></a>
<figcaption><b>{html.escape(os.path.basename(k))}</b><br>{v['w']}x{v['h']} &middot; {v['font']}<br>
{v['chars']:,} chars &middot; {v['claude_tokens']:,} tok &middot;
<b>{v['chars_per_claude_token']:.1f}</b> ch/tok</figcaption></figure>""")
write_output('index.html', f"""<!doctype html><meta charset=utf-8>
<title>Dense text in images</title>
<style>
body{{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1{{font-size:20px}} main{{display:grid;gap:18px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}}
figure{{margin:0;background:#1c1c1f;border:1px solid #333;border-radius:6px;padding:8px}}
img{{width:100%;image-rendering:pixelated;background:#fff;border-radius:3px}}
figcaption{{font-size:11px;color:#aaa;margin-top:6px;font-family:ui-monospace,monospace}}
</style><h1>Dense text in images &mdash; {len(E)} test images</h1>
<p style="color:#999">Thumbnails scaled; click for 1:1. Claude 28px/token, GPT-5.6 32px/token.</p>
<main>{''.join(cards)}</main>""")
print(f"README.md ({len(S)} lines) + index.html written")

# ---- Part II: the GPT-5.6 / Sol line
import glob as _g3, statistics as _st3
geo = {}
for f in sorted(_g3.glob(os.path.join(ROOT, 'results_geometry_*_gpt-5.6-*.json'))):
    b = json.load(open(f)); geo.setdefault(b['model'], {})[b.get('patch')] = b
conf56 = {}
for f in sorted(_g3.glob(os.path.join(ROOT, 'results_confirm-v1_gpt-5.6-*.json'))):
    b = json.load(open(f)); conf56[b['manifest']['model']] = b['results']
if geo or conf56:
    BASE = {'clR5x10', '6x13', '8x16'}
    L = ["", "---", "", "# Part II - the GPT-5.6 (Sol) line", "",
         "Run through `codex exec` against `gpt-5.6-sol`, `gpt-5.6-luna` and `gpt-5.6-terra`,",
         "reasoning effort pinned `low`, on the **same images** the Claude arm used. 106 of the",
         "128 images already have both dimensions divisible by 32 - a side effect of building the",
         "corpus on multiples of 224 = lcm(28,32) - so both providers see byte-identical stimuli",
         "with zero quantisation waste on either grid.", "",
         "## 15. GPT-5.6 geometry", "",
         "32x32 patches confirmed - **but a patch does not cost one token.** Fitting billed input",
         "against patch count across 224/448/896/1344 squares:", "",
         "    tokens = base + 1.2 x patches", "",
         "| canvas | patches | sol tokens | implied base at 1.2 |", "|---|---|---|---|",
         "| 224x224 | 49 | 14,048 | 13,989.2 |", "| 448x448 | 196 | 14,225 | 13,989.8 |",
         "| 896x896 | 784 | 14,930 | 13,989.2 |", "| 1344x1344 | 1,764 | 16,108 | 13,991.2 |", "",
         "The base is constant to within 2 tokens across a 36x range in patch count, so the 1.2",
         "multiplier is real, not noise. Effective rate: **1024/1.2 = 853 px per billed token**",
         "against Claude's 784 - a uniform **+8.8% density advantage at every cell**.", "",
         "### The ceiling is a patch cap at 1600x1600", "",
         "1792, 2240, 2688 and 3200 pixel squares **all bill exactly 16,992**. Bisecting:", "",
         "| canvas | patches | measured | uncapped prediction | verdict |", "|---|---|---|---|---|",
         "| 1536x1536 | 2,304 | 16,756 | 16,755 | uncapped |",
         "| 1600x1600 | 2,500 | 16,992 | 16,990 | at the cap |",
         "| 1632x1632 | 2,601 | 16,992 | 17,111 | **capped** |",
         "| 1664x1664 | 2,704 | 16,992 | 17,235 | **capped** |", "",
         "So anything past 1600px is downscaled back to 50x50 = 2,500 patches. Unlike Claude's",
         "ceiling, this one needed no paired deltas or min-of-N: **GPT-5.6 token accounting is",
         "deterministic**, returning identical values across reps at every size.", "",
         "### Family-wide, not model-specific", "",
         "| model | tokens/patch (896->1600) | base prompt | cap |", "|---|---|---|---|",
         "| gpt-5.6-sol | 1.2016 | 13,990 | 1600px |",
         "| gpt-5.6-luna | 1.2016 | 12,426 | 1600px |",
         "| gpt-5.6-terra | 1.2016 | 13,990 | 1600px |", "",
         "Identical to four decimal places; only the base prompt length differs (system-prompt",
         "size, not geometry). All three accept attached images.", "",
         "## 16. Capacity: the two providers trade off in opposite directions", "",
         "| | patch | tok/patch | px/token | max square | image tokens |", "|---|---|---|---|---|---|",
         "| claude | 28 | 1.0 | 784 | 1932x1932 | 4,761 |",
         "| gpt-5.6 | 32 | 1.2 | **853** | **1600x1600** | 3,000 |", "",
         "One maximum-size page:", "",
         "| cell | claude chars | claude ch/tok | gpt-5.6 chars | gpt-5.6 ch/tok |",
         "|---|---|---|---|---|",
         "| 5x10 | **74,498** | 15.65 | 51,200 | **17.07** |",
         "| 6x13 | **47,656** | 10.01 | 32,718 | **10.91** |",
         "| 8x16 | **28,920** | 6.07 | 20,000 | **6.67** |", "",
         "GPT-5.6 is denser **per token**; Claude holds more **per image**. Optimising cost per",
         "character picks GPT-5.6; optimising characters per request picks Claude.", ""]
    if conf56:
        L += ["## 17. Fidelity on identical images", "",
              "`confirm.py` base matrix - 5x10 / 6x13 / 8x16 x 2 reps, 56-character records:", "",
              "| model | payload | decode exact | mean CER | returned the asked-for row |",
              "|---|---|---|---|---|"]
        for m in sorted(conf56):
            R = conf56[m]
            for pl in ['prose', 'alnum_easy', 'hex', 'b64']:
                d = [r for r in R if r['payload'] == pl and r['probe'] == 'decode' and r['font'] in BASE]
                bd = [r for r in R if r['payload'] == pl and r['probe'] == 'bind' and r['font'] in BASE]
                if not d: continue
                L.append(f"| {m} | {pl} | {sum(r['exact'] for r in d)}/{len(d)} | "
                         f"{_st3.mean(r['cer'] for r in d):.3f} | "
                         f"{sum(1 for r in bd if r.get('displacement') == 0)}/{len(bd)} |")
        L += ["", "### Against Claude, on the metric that mattered most", "",
              "High-entropy exact decode of a 56-character record - the number that was **0/18** on",
              "Claude at every cell and alphabet:", "",
              "| model | exact | rate | 95% CI | paired McNemar vs Claude |", "|---|---|---|---|---|",
              "| opus (claude) | 0/18 | 0.00 | [0.00, 0.15] | - |",
              "| gpt-5.6-luna | 2/18 | 0.11 | [0.02, 0.31] | 0.500 |",
              "| gpt-5.6-sol | 3/18 | 0.17 | [0.05, 0.38] | 0.250 |",
              "| gpt-5.6-terra | 4/18 | 0.22 | [0.08, 0.44] | 0.125 |",
              "| 5.6 pooled | 9/54 | 0.17 | [0.09, 0.27] | - (not one paired model) |", "",
              "**No individual comparison is significant.** Every 5.6 model recovered records where",
              "Claude recovered none, and the direction is consistent across all three, but the CIs",
              "overlap and the pooled figure mixes 3 models x 3 cells x 3 payloads - the same",
              "heterogeneous pooling criticised in section 5. The supportable claim is *a consistent",
              "directional difference that n=18 cannot establish*, not a demonstrated advantage.", "",
              "### Binding is where sol clearly separates", "",
              "| model | returned the asked-for row | bind exact |", "|---|---|---|",
              "| opus (claude) | 6/24 | 3/24 |", "| gpt-5.6-luna | 9/24 | 1/24 |",
              "| **gpt-5.6-sol** | **17/24** | 6/24 |", "| gpt-5.6-terra | 12/24 | 1/24 |", "",
              "Row addressing was the dominant Claude failure and the reason E-BIND-1 routes around",
              "it entirely. Sol hits the requested row 71% of the time against Claude's 25%, and",
              "that gap separates sol from its own siblings as much as from Claude.", "",
              "### Bigger glyphs DO fix exactness on GPT-5.6 - a Claude-specific finding, corrected", "",
          "Section 3 concluded that enlarging glyphs does not eliminate exact-copy failures: Claude",
          "managed 1/12 on whole 56-character records even at 9x18/10x20/12x24 (up to 288 px per",
          "character). That is **not** a general property of vision models - it is a property of",
          "Claude. On the same images:", "",
          "| model | decode exact (larger cells) | mean CER |", "|---|---|---|",
          "| **gpt-5.6-terra** | **8/12** | 0.009 |", "| gpt-5.6-sol | 6/12 | 0.098 |",
          "| gpt-5.6-luna | 4/12 | 0.025 |", "| opus | 1/12 | 0.046 |", "",
          "So the remedy Part I ruled out - just render bigger - works on GPT-5.6 and not on",
          "Claude. Note also that terra is the *worst* of the three at small-glyph legibility and",
          "the *best* at exact transcription of large ones, so these are separable capabilities",
          "and a single model ranking would hide that.", "",
          "### Inversion worth keeping", "",
              "On **prose** Claude was better (5/6 against sol's 4/6). The 5.6 advantage is specific",
              "to unguessable payload - consistent with less reliance on language priors to repair",
              "ambiguous glyphs, which is exactly the redundancy credit measured in section 3.", "",
              "### Caveats attached to every number above", "",
              "* **Image delivery differs.** `codex exec` attaches the image with `-i`; `claude -p`",
              "  receives a path and fetches it with the Read tool. Stimuli identical, path not.",
              "  This also means the shell-out-and-zoom confound is structurally impossible on",
              "  codex, whereas Claude needed explicit tool disallowing.",
              "* **Effort is pinned `low`** for the 5.6 models (their default). The Claude runs had",
              "  no equivalent control, so the comparison is not effort-matched.",
              "* **n = 18 per model** across mixed cells and payloads.", ""]
    readme += chr(10).join(L) + chr(10)
    write_output('README.md', readme)

# ---- Part II continued: ladder / legibility / span / delim on the 5.6 line
import glob as _g4, statistics as _st4
def _lb(k, n):
    try:
        from scipy.stats import beta
        return beta.ppf(0.05, k, n - k + 1) if k > 0 else 0.0
    except ImportError:
        return float('nan')
lad56  = {os.path.basename(f).replace('results_ladder_','').replace('.json',''): V.result_rows(json.load(open(f)))
          for f in sorted(_g4.glob(os.path.join(ROOT, 'results_ladder_gpt-5.6-*.json')))}
edge56 = {os.path.basename(f).replace('results_edge_','').replace('.json',''): V.result_rows(json.load(open(f)))
          for f in sorted(_g4.glob(os.path.join(ROOT, 'results_edge_gpt-5.6-*.json')))}
span56 = {os.path.basename(f): V.result_rows(json.load(open(f)))
          for f in sorted(_g4.glob(os.path.join(ROOT, 'results_span_6x13_*_gpt-5.6-*.json')))}
del56  = {os.path.basename(f): V.result_rows(json.load(open(f)))
          for f in sorted(_g4.glob(os.path.join(ROOT, 'results_delim_6x13_gpt-5.6-*.json')))}
L = []
if lad56:
    L += ["", "## 18. The bitmap ladder, same images as Claude", "",
          "`run_eval.py` over series A. *lookup* = exact match on the code at a named line;",
          "*legible* = similarity to whichever ground-truth line the answer best matches.", "",
          "| glyph | ch/token (28-grid) | " + ' | '.join(f"{m} lookup | {m} legible" for m in sorted(lad56)) + " |",
          "|---|---|" + "---|" * (2 * len(lad56))]
    order = None
    for m in sorted(lad56):
        rows = {r['glyph']: r for r in lad56[m]}
        if order is None: order = sorted(rows, key=lambda g: -rows[g]['ch_per_tok'])
    for g in order or []:
        line = f"| {g} | {list(lad56.values())[0][0]['ch_per_tok'] if False else ''} | "
        first = True
        cells = []
        dens = None
        for m in sorted(lad56):
            r = {x['glyph']: x for x in lad56[m]}.get(g)
            if r is None: cells += ['-', '-']; continue
            dens = r['ch_per_tok']
            cells.append('-' if r.get('code_acc') is None else f"{r['code_acc']*100:.0f}%")
            cells.append('-' if r.get('legibility') is None else f"{r['legibility']:.2f}")
        L.append(f"| {g} | {dens:.1f} | " + ' | '.join(cells) + " |")
    L.append("")
if edge56:
    _all = {'opus': 'results_edge_opus.json'}
    for _m in sorted(edge56): _all[_m] = f'results_edge_{_m}.json'
    _rows = {}
    for _m, _f in _all.items():
        try: _d = V.result_rows(json.load(open(os.path.join(ROOT, _f))))
        except Exception: continue
        for r in _d: _rows.setdefault(r['file'], {})[_m] = r
    _models = [m for m in _all if any(m in v for v in _rows.values())]
    _shared = [k for k, v in _rows.items() if all(m in v for m in _models)]
    _shared.sort(key=lambda k: -_rows[k][_models[0]]['ch_per_tok'])
    if _shared and len(_models) > 1:
        L += ["## 19. Legibility isolated from addressing", "",
              "First and last lines only - they sit at the page edges, so no row counting is",
              "involved and the score is glyph resolution alone. Byte-identical files across all",
              f"models, n={len(_shared)}.", "",
              "| cell | ch/token | " + ' | '.join(_models) + " |",
              "|---|---|" + "---|" * len(_models)]
        for k in _shared:
            r0 = _rows[k][_models[0]]
            L.append(f"| {r0['cell']} | {r0['ch_per_tok']:.1f} | " +
                     ' | '.join(f"{_rows[k][m]['legibility']:.2f}" for m in _models) + " |")
        L += ["", "| model | mean legibility |", "|---|---|"]
        for m in sorted(_models, key=lambda m: -_st4.mean(_rows[k][m]['legibility'] for k in _shared)):
            L.append(f"| {m} | **{_st4.mean(_rows[k][m]['legibility'] for k in _shared):.3f}** |")
        L += ["", "Three bands. Below ~8 ch/token every model scores 0.94-1.00 and nothing is",
              "distinguishable. Between 9.7 and 15.2 sol dominates while opus is erratic (0.00 to",
              "0.99 across neighbouring conditions) and terra/luna are weak. Above 16.9 only sol",
              "and opus read at all - and **sol alone reads anything at 21.3 ch/token** (0.39",
              "where every other model scores 0.00).", ""]
if span56:
    import re as _re
    _nn = lambda x: _re.sub(r'\s+', '', str(x))
    def _pfx(r): return r.get('prefix', r['exact'])
    L += ["## 20. Short spans at n=20 - two metrics, because they measure different things", "",
          "`exact` requires the answer to be exactly the N characters requested. `prefix` asks",
          "only whether the answer *starts* with them. The gap is not a reading difference: sol",
          "frequently returns N+1 characters, completing the visible 4-character group rather than",
          "cutting mid-group. Opus scores identically on both metrics (49/80 either way) - it",
          "always cuts at exactly N - so the split isolates **instruction-following** from",
          "**reading**.", "",
          "| model | alphabet | metric | span 8 | span 16 | span 32 | span 51 | total |",
          "|---|---|---|---|---|---|---|---|"]
    claude_hex = None
    fp = os.path.join(ROOT, 'results_span_6x13_hex_opus.json')
    if os.path.exists(fp): claude_hex = V.result_rows(json.load(open(fp)))
    def _row(label, alpha, d, key):
        cells, tot, totn = [], 0, 0
        for n in (8, 16, 32, 51):
            rs = [r for r in d if r['span'] == n]
            k = sum(key(r) for r in rs)
            cells.append(f"{k}/{len(rs)}" if rs else '-')
            tot += k; totn += len(rs)
        return f"| {label} | {alpha} | {'literal' if key is not _pfx else '**prefix**'} | " + \
               ' | '.join(cells) + f" | **{tot}/{totn}** |"
    if claude_hex:
        L.append(_row('opus (claude)', 'hex', claude_hex, lambda r: r['exact']))
    for fn in sorted(span56):
        d = span56[fn]
        mdl = fn.replace('results_span_6x13_', '').replace('.json', '')
        alpha, model = mdl.split('_', 1)
        L.append(_row(model, alpha, d, lambda r: r['exact']))
        L.append(_row(model, alpha, d, _pfx))
    L += ["", "**Alphabet dominates model.** Crockford base32 fails for everyone at 6x13 -",
          "opus 20/80, sol 17/80, luna 15/80, no meaningful separation - while hex separates the",
          "models cleanly (sol 73/80, luna 55/80, opus 49/80). Whatever makes uppercase base32",
          "hard at this size defeats every model tested, so alphabet choice is a bigger lever",
          "than model choice.", "",
          "**Do not read hex vs crock32 here as an alphabet comparison.** 32 crock32 symbols",
          "carry 160 bits against hex's 128, so the crock32 task is simply harder - the same",
          "equal-character-count confound identified in section 3. The delimited probe (section 21)",
          "holds *information* constant and is the clean test. What this table does show is that",
          "crock32 prefix-tolerance rescues nothing (17 -> 17) whereas hex gains 18, so crock32",
          "failures are genuine misreads rather than boundary artifacts.", "",
          "Two results stand out. At **span 51 Claude scores 0/20** - a hard failure regime I",
          "attributed to counting, boundary tracking or output stamina (section 4) - while sol",
          "reads it 16/20. And at **span 32 sol is 20/20 prefix-correct**: a 128-bit handle read",
          "byte-exact, twenty times out of twenty, where Claude managed 15/20.", ""]
if del56:
    L += ["## 21. Delimited equal-information fields at n=20", "",
          "| model | bits | alphabet | chars | literal | value | 95% LB |",
          "|---|---|---|---|---|---|---|"]
    for fn in sorted(del56):
        d = del56[fn]; model = fn.replace('results_delim_6x13_', '').replace('.json', '')
        for b in (64, 128):
            for e in ('hex', 'b32'):
                rs = [r for r in d if r['bits'] == b and r['enc'] == e]
                if not rs: continue
                v = sum(r['value'] for r in rs)
                L.append(f"| {model} | {b} | {e} | {rs[0]['chars']} | "
                         f"{sum(r['literal'] for r in rs)}/{len(rs)} | **{v}/{len(rs)}** | "
                         f"{_lb(v, len(rs)):.3f} |")
    L.append("")
if L:
    readme += chr(10).join(L) + chr(10)
    write_output('README.md', readme)

# ---- Part II conclusions
try:
    import subprocess as _sp
    _cmp = _sp.run([sys.executable, os.path.join(ROOT, 'compare.py')],
                   capture_output=True, text=True, timeout=120).stdout
except Exception:
    _cmp = ''
if _cmp.strip():
    L = ["", "## 22. Cross-provider conclusions", "",
         "Generated by `compare.py` - every table is paired on identical images.", "",
         "```", _cmp.rstrip(), "```", "",
         "### What holds up", "",
         "* **Sol reads dense glyphs better than Opus - but its siblings do not.** 26 paired",
         "  legibility conditions on byte-identical files, addressing removed:", "",
         "  | model | opus -> model | delta | Wilcoxon p |", "  |---|---|---|---|",
         "  | **gpt-5.6-sol** | 0.606 -> **0.761** | **+0.155** | **0.0094** |",
         "  | gpt-5.6-luna | 0.606 -> 0.480 | -0.126 | 0.0296 |",
         "  | gpt-5.6-terra | 0.606 -> 0.426 | -0.180 | 0.0266 |", "",
         "  Only sol beats Claude; **luna and terra are both significantly worse**. Any claim of",
         "  the form 'GPT-5.6 reads dense text better' is false as a family statement - the",
         "  geometry is family-wide, the capability is not.",
         "* **The long-span failure regime is Claude-specific.** Claude scored 0/20 at a",
         "  51-character span - a cliff attributed in section 4 to counting or output stamina.",
         "  Both 5.6 models tested read it: sol 16/20, luna 12/20. So that cliff is a property of",
         "  Claude, not of vision models, and section 4's 'separate failure regime' should be read",
         "  as provider-specific. Sol also hits 20/20 prefix-correct at 32 characters - a 128-bit",
         "  handle read byte-exact, twenty times out of twenty.",
         "* **Sol binds rows far better.** 17/24 against Claude's 6/24 on the same probe. Row",
         "  addressing was *the* blocker on the Claude side and the entire reason E-BIND-1 routes",
         "  around it.",
         "* **The advantage is specific to unguessable payload.** On prose Claude is equal or",
         "  better. That is consistent with the redundancy-credit finding in section 3: Claude",
         "  leans on language priors to repair glyphs, and those priors do nothing for a hash.", "",
         "### What does not", "",
         "* **The whole-record comparison is underpowered.** 3/18 against 0/18 sounds decisive and",
         "  is not: paired McNemar p = 0.250, CIs overlap, and pooling 3 cells x 3 payloads is exactly the",
         "  heterogeneous pooling criticised in section 5. The span and legibility probes at n=20",
         "  carry the argument; the confirm matrix does not.",
         "* **Sol follows length instructions worse.** Its literal span score is 55/80 against",
         "  Claude's 49/80, but only because prefix-tolerance rescues 18 answers where it returned",
         "  N+1 characters. Claude cuts at exactly N every time. If your parser is strict, that",
         "  difference is real and costs you.",
         "* **Two of the three 5.6 models are worse than Claude at this task.** Ladder lookup:",
         "  sol 55%, luna 37%, terra 32% against opus 39% and sonnet 62%. Picking 'the 5.6 line'",
         "  without picking sol specifically would be a downgrade.", "",
         "### Choosing between them", "",
         "| if you care about | pick | why |", "|---|---|---|",
         "| cost per character | **gpt-5.6** | 853 vs 784 px/token, +8.8% at every cell |",
         "| characters per request | **claude** | 1932x1932 page holds 74,498 vs 51,200 |",
         "| reading very dense glyphs | **gpt-5.6-sol** | legibility 0.761 vs 0.606; reads 21.3 ch/token where nothing else does |",
         "| **recovering an exact handle** | **claude** | delimited fields 70/80 vs sol 53/80, luna 22/80 |",
         "| **resolving the right record** | **unresolved** | Claude leads a debug-only 20-item preflight; only sol has Stage 1 |",
         "| **knowing when there is no answer** | **sol is ruled out** | sol false-accepts 36.7% in Stage 1; Claude's 0/5 is underpowered |",
         "| exact adherence to output format | **claude** | cuts at exactly N; sol over-returns |",
         "| latency | **gpt-5.6** | 20 s/image against 43 (opus) and 104 (sonnet) |", "",
         "**The rankings cannot yet be compared at confirmatory scale.** Sol reads glyphs better",
         "than Claude and fails the Stage-1 abstention requirement. Claude leads the small preflight,",
         "but its protocol safety and resolution rates need the same held-out Stage-1 design before",
         "it can be called the winner.", "",
         "Neither is safe for unverified handles. Every conclusion in Part I about checksums,",
         "canonical fetch and provenance binding applies unchanged to both providers - and sol's",
         "no-answer false-accept rate makes post-fetch reconciliation *more* necessary, not less.", ""]
    readme += chr(10).join(L) + chr(10)
    write_output('README.md', readme)

# ---- E-BIND-1 across providers
import glob as _g5, collections as _c5
eb = {}
for f in sorted(_g5.glob(os.path.join(ROOT, 'results_ebind1_*.json'))):
    b = json.load(open(f))
    eb[(b.get('model', '?'), b.get('stage', '?'))] = b
if eb:
    L = ["", "## 23. E-BIND-1: resolving the correct canonical record", "",
         "Semantic query, exact text page/block labels outside the bitmap, a 64-bit keyed",
         "codeword inside it, canonical fetch and gold scoring. Outcome partitions are separate",
         "for answer-present (`C`/`D`/`W`/`A`/`P`) and no-answer (`N`/`W0`/`D0`/`P0`) queries,",
         "because abstention is correct in one and failure in the other.", "",
         "| model | stage | carrier | C | D | **W** | A | N | **W0** | D0 | P |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (mdl, stage), b in sorted(eb.items()):
        for carrier, encf in (('image', 'hex'), ('image', 'b32'), ('text', 'hex')):
            rs = [r for r in b['results']
                  if r.get('carrier') == carrier and (r.get('enc') or 'hex') == encf]
            if not rs: continue
            carrier = f"{carrier}/{encf}"
            c = _c5.Counter(r['outcome'] for r in rs)
            L.append(f"| {mdl} | {stage} | {carrier} | {c.get('C',0)} | {c.get('D',0)} | "
                     f"**{c.get('W',0)}** | {c.get('A',0)} | {c.get('N',0)} | "
                     f"**{c.get('W0',0)}** | {c.get('D0',0)} | {c.get('P',0)+c.get('P0',0)} |")
    # false-accept rates with one-sided bounds - the load-bearing safety number
    L += ["", "### The preflight text ceiling isolates optics from semantics", "",
          "Every model scores **20/20 C and 5/5 N on the text carrier** - identical corpus,",
          "identical queries, identical distractors, records supplied as text instead of pixels.",
          "So semantic selection, abstention and protocol compliance are all intact in all of",
          "them, and the entire spread on the image carrier (opus 20, sol 17, luna 1) is optical.",
          "Without this control a score of 1/20 could not be separated into 'cannot read' versus",
          "'cannot find', and luna would look semantically broken when it is not.", "",
          "The preflight also separates *reading* from *calibration* cleanly: terra (8/20 C) and",
          "luna (1/20 C) read much worse than sol (17/20 C) yet both abstain correctly 5/5, while",
          "sol abstains 0/5. Confidence and competence are independent here.", "",
          "Note the preflight ranking inverts against legibility: sol reads glyphs better than opus",
          "(section 19) yet opus leads this small protocol preflight, which additionally requires",
          "exact code transcription, correct block selection and correct abstention - the three",
          "things sol is weaker at.", "",
          "### False-accept rates with call-level bounds", "",
          "The Stage-1 bound in this table treats repeated calls as independent and is retained only",
          "for historical comparability; the clustered interval below is the inferential result.", "",
          "| model | stage | carrier | present n | **W** | W rate | call-level 95% UB | no-answer n | **W0** | W0 rate |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (mdl, stage), b in sorted(eb.items()):
        for carrier in ('image', 'text'):
            pres = [r for r in b['results']
                    if r.get('carrier') == carrier and r.get('kind') == 'present'
                    and r.get('enc') in (None, 'hex')]
            if not pres: continue
            w = sum(1 for r in pres if r['outcome'] == 'W')
            try:
                from scipy.stats import beta as _bt
                ub = _bt.ppf(0.95, w + 1, len(pres) - w)
            except Exception:
                ub = float('nan')
            noa = [r for r in b['results']
                   if r.get('carrier') == carrier and r.get('kind') == 'absent'
                   and r.get('enc') in (None, 'hex')]
            w0 = sum(1 for r in noa if r['outcome'] == 'W0')
            L.append(f"| {mdl} | {stage} | {carrier} | {len(pres)} | **{w}** | "
                     f"{w/len(pres):.3f} | {ub:.3f} | {len(noa)} | **{w0}** | "
                     f"{(w0/len(noa)) if noa else 0:.3f} |")
    _s1 = eb.get(('gpt-5.6-sol', 'stage1'))
    if _s1:
        import collections as _c6
        _R = _s1['results']
        _items = _c6.defaultdict(list)
        for r in _R: _items[r['query']].append(r['outcome'])
        _consistent = sum(1 for v in _items.values() if len(set(v)) == 1)
        _witems = sum(1 for v in _items.values() if 'W' in v)
        L += ["", "### Stage 1 at scale, and why the bounds above are optimistic", "",
              "360 calls on 12 blocks / 144 records. Answer-present: **C 63.3%**, D 34.0%,",
              "**W 2.7%**, A 0%. No-answer: **N 3.3%**, **W0 36.7%**, D0 60%.", "",
              f"But those 360 calls are **{len(_items)} unique items x 3 passes**, not 360",
              f"independent trials. {_consistent}/{len(_items)} items ({_consistent/len(_items):.0%}) "
              "returned the identical outcome on all three passes, and only",
              f"**{_witems} distinct items** ever produced a `W`. The repeated passes estimate",
              "call-level variance conditional on the item set; they do not multiply the item",
              "sample. Per-item n is " + str(len(_items)) + ", so every bound in the table above is",
              "optimistic - the correct denominator for generalising to new items is the item",
              "count, not the call count.", "",
              "Item/page-clustered bootstrap intervals (rather than treating 360 calls as independent)",
              "put C at 0.633 [0.533, 0.725], W at 0.027 [0.000, 0.065], and W0 at",
              "0.367 [0.217, 0.517].", "",
              "### The stored Stage-1 text ceiling is unpaired", "",
              "The text run used the `stage1text` seed namespace while the image run used `stage1`;",
              "their query sets have zero overlap. It is a useful semantic ceiling on a different",
              "120-item corpus, not the claimed same-item carrier control:", "",
              "| carrier | resolution | correct abstention |", "|---|---|---|",
              "| text | **99.0%** (99/100) | **100%** (20/20) |",
              "| image, hex | 63.3% (190/300) | 3.3% (2/60) |", "",
              "The contrast is large and shows that sol can execute the protocol from text, but the",
              "old Fisher tests treated repeated image calls as independent and compared different",
              "items. They are withdrawn. The new held-out campaign supplies both carriers from one",
              "frozen corpus and analyzes their difference by unique item/page.", "",
              "**Sol never abstains on the image carrier.** `A` = 0 across all 300 answer-present",
              "calls and `N` = 2/60 on no-answer. It always produces something; the keyed tag then",
              "rejects 34% of it. On more than a third of questions with no answer at all, sol",
              "returns a validating codeword for a real - but wrong - record. That failure is",
              "invisible to any checksum.", ""]
    L += ["", "A `W` is a *valid* codeword for a *real* record, returned for a different record's",
          "question. No checksum or tag width detects it - only reconciliation after canonical",
          "fetch. This is the number that decides whether an optical lookup tier is safe.", "",
          "### Sol's abstention breaks only on the image carrier", "",
          "Sol returned `N` (correct NO_MATCH) **5/5** on the text carrier and **0/5** on the",
          "image carrier, where all five no-answer queries produced something instead: 3 `W0` plus",
          "2 `D0`. Claude abstained correctly in both. So this is not a general calibration",
          "failure - sol knows when text contains no answer, and stops knowing when the same",
          "records arrive as pixels. That is the dangerous direction, because `W0` is a *valid*",
          "codeword for a *real* record returned for a question with no answer.", "",
          "Sol's base32 collapse also reappears here (4/20 against Claude's 17/20) - the third",
          "independent confirmation after the span and delimited-field probes.", "",
          "`W` is the load-bearing number: a wrong canonical record accepted as correct. `W0`",
          "is its no-answer twin - a real record returned for a query that has no answer. A",
          "checksum cannot catch either, because the returned codeword is a valid codeword for a",
          "real record; only reconciliation after fetch can.", ""]
    readme += chr(10).join(L) + chr(10)
    write_output('README.md', readme)

# ---- sol-native 32-grid frontier
_edge_sol = _load_json(os.path.join(ROOT, 'results_edge_gpt-5.6-sol.json')) if False else None
try:
    _edge_sol = V.result_rows(json.load(open(os.path.join(ROOT, 'results_edge_gpt-5.6-sol.json'))))
except Exception:
    _edge_sol = None
if _edge_sol:
    p32 = [r for r in _edge_sol if '_p32' in r['file']]
    if p32:
        L = ["", "## 24. Sol-native frontier (32-grid canvases)", "",
             "Sections 18-21 deliberately reuse Claude's 28-grid images so the stimuli are",
             "identical. Those canvases are not optimal for a 32px patch grid, so this section",
             "regenerates the packing series on zero-waste 32-grid canvases and re-measures. The",
             "`ch/token` column here is **sol's own** rate (1024/1.2 per patch), not Claude's.", "",
             "| cell | canvas | ch/token (sol) | legibility |", "|---|---|---|---|"]
        seen = {}
        for r in sorted(p32, key=lambda r: -r['ch_per_tok']):
            v = E.get(r['file'], {})
            key = (r['cell'], v.get('w'), v.get('h'))
            if key in seen: continue
            seen[key] = 1
            L.append(f"| {r['cell']} | {v.get('w')}x{v.get('h')} | "
                     f"{v.get('chars_per_gpt_token', float('nan')):.1f} | {r['legibility']:.2f} |")
        good = [r for r in p32 if r['legibility'] >= 0.95]
        if good:
            best = max(good, key=lambda r: E.get(r['file'], {}).get('chars_per_gpt_token', 0))
            vb = E.get(best['file'], {})
            L += ["", f"Densest 32-grid cell reaching legibility >= 0.95: **{best['cell']}** at "
                      f"**{vb.get('chars_per_gpt_token', 0):.1f} chars per sol token** "
                      f"({vb.get('w')}x{vb.get('h')}, legibility {best['legibility']:.2f}).", ""]
        readme += chr(10).join(L) + chr(10)
        write_output('README.md', readme)

# ---- Part II method + final state
L = ["", "---", "", "## 25. Method: the GPT-5.6 arm", "",
     "Every 5.6 call goes through `providers.py`, which normalises two genuinely different",
     "CLIs behind one interface:", "",
     "```",
     "codex exec --ephemeral --ignore-user-config --skip-git-repo-check \\",
     "           -s read-only -m <exact-slug> -c model_reasoning_effort=low \\",
     "           -i <image> --json          # prompt on STDIN, never as a positional arg",
     "```", "",
     "| flag | why it is mandatory |", "|---|---|",
     "| `--ignore-user-config` | the local `config.toml` sets `model_reasoning_effort = high` and a personality; without this every run is silently a different experiment |",
     "| `-c model_reasoning_effort=low` | effort is a free variable the Claude arm never controlled; pinned and recorded |",
     "| `--ephemeral` | no session carry-over between trials |",
     "| `-s read-only` | no side effects |",
     "| `-i <image>` | the image is **attached**, so the model never receives a path |",
     "| prompt on stdin | `-i` is variadic and eats a trailing positional prompt |", "",
     "**The one asymmetry that cannot be removed.** `codex exec` attaches images; `claude -p`",
     "receives a path and fetches with the Read tool. Stimuli are byte-identical, delivery is",
     "not. A side effect is that the crop-and-zoom confound which forced `--disallowedTools` on",
     "the Claude side is *structurally impossible* on codex - there is no path to act on - so",
     "the codex arm is cleaner in that one respect and different in another. Every comparison",
     "table says so.", "",
     "Each result records the exact model slug (never `sol`/`opus`), effective effort (`null` for",
     "uncontrolled Claude runs), CLI version, prompt",
     "SHA and image-delivery mode. Usage comes from the `turn.completed` event of the `--json`",
     "stream - note that session files use a different shape (`token_count`), which cost one",
     "debugging cycle.", "",
     "### Call budget actually spent", "",
     "| phase | calls |", "|---|---|",
     "| 1 geometry (3 models, rate + ceiling + bisect + adjacent) | ~90 |",
     "| 2 ladder + legibility + confirm + extension (3 models) | ~460 |",
     "| 3 spans hex + crock32 + delimited (3 models) + sol-native p32 | ~750 |",
     "| 4 E-BIND-1 preflight (3 models) + Stage 1 (hex x3, b32, text) | ~800 |",
     "| **total** | **~2,100** |", "",
     "The plan budgeted ~2,230 on Claude timings; 5.6 calls run 5-20s against Claude's 10-280s,",
     "so wall-clock was a fraction of the estimate.", "",
     "---", "", "## 26. Where this leaves things", "",
     "**Solved.** Image-token geometry for both providers, zero-waste canvas construction, the",
     "downscale ceilings (Claude 1932x1932 at 4,761 patches; GPT-5.6 1600x1600 at 2,500 patches",
     "billing 1.2 tokens each), and the density frontier for every glyph cell on both grids.", "",
     "**Measured, with the caveats stated.** Legibility, span reading and delimited-field recovery",
     "cover five models on byte-identical stimuli. Full Stage-1 protocol resolution exists only",
     "for sol; the other models have debug preflights and must not yet be ranked as protocol winners.", "",
     "**Established and unwelcome.** No configuration on either provider is safe for unverified",
     "handles. Sol additionally returns a valid codeword for a real record on ~37% of Stage-1",
     "no-answer calls, which no checksum can detect. Post-fetch semantic reconciliation is mandatory",
     "for sol; other providers require held-out estimates before any safety comparison.", "",
     "**Implemented but not yet executed.** The 2,265-call campaign includes a 32-bit index +",
     "32-bit keyed tag, paired image/text carriers, Arm-B native-grid goodput, fail-closed",
     "verification of deliberately wrong-valid fetched records, the 2x2 grouping-by-delimiting",
     "design, declared-margin equivalence with boundary-safe discordant-cell intervals, larger",
     "cells, effort sweeps and rectangular geometry. Paid runs remain explicit through",
     "`campaign.py --execute`; `--resume` skips only non-dry artifacts with the expected row count.", "",
     "The next measurement is the held-out structured-codeword Stage 1 on the actual deployment",
     "candidate, followed by Arm B and the verifier arm. Those estimates, not characters per",
     "token or a debug preflight,",
     "decides whether an optical memory tier is viable.", ""]
readme += chr(10).join(L) + chr(10)
write_output('README.md', readme)

if CHECK:
    different = False
    for name in ('README.md', 'index.html'):
        expected_path = os.path.join(ROOT, name)
        generated_path = os.path.join(OUTPUT_ROOT, name)
        expected = open(expected_path).read().splitlines(keepends=True)
        generated = open(generated_path).read().splitlines(keepends=True)
        if expected != generated:
            different = True
            sys.stderr.writelines(difflib.unified_diff(
                expected, generated, fromfile=name, tofile=f'generated/{name}', n=3))
    shutil.rmtree(OUTPUT_ROOT)
    if different:
        raise SystemExit('generated documentation is stale; run python3 make_docs.py')
    print('generated documentation: up to date')
