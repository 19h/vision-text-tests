# Dense text in images

How many characters fit in one image token, and how many of them come back correctly.
128 generated test images, 392 graded model runs, and a
record of which conclusions survived scrutiny.

Two separate questions, with very different answers:

* **Capacity** is solved and well measured. A 1932x1932 page holds 74,498 characters of
  prose at 15.7 characters per image token.
* **Fidelity** is not. Arbitrary data comes back byte-exact only in short delimited
  fields, and only some of the time. Nothing measured here is safe to trust unverified.

| | |
|---|---|
| Pixels per image token | **784** (28 x 28), confirmed by billing deltas |
| Largest un-downscaled square | **1932 x 1932** (69 x 69 = 4,761 patches) |
| Prose capacity, one image | **74,498 chars** at 5x10, 15.7 chars/token |
| Best observed field | **20/20** on a delimited 64-bit hex field (candidate, post-selection) |
| Exact recovery of a 56-char record | **0/18** in the base matrix; 1/30 including larger cells |
| Record binding by row number | **broken** - 0/6 for every high-entropy payload |

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
similarity score and a useless hash.

### Whole records (56 chars, `confirm.py`)

Base matrix: 5x10, 6x13, 8x16 x 2 reps. Conditions are heterogeneous, so per-payload
rows are shown separately rather than pooled into one rate.

| payload | alphabet | decode exact | mean CER | bind exact | bind hit a real row |
|---|---|---|---|---|---|
| prose | closed-vocabulary English | **5/6** | 0.003 | 3/6 | 4/6 |
| alnum_easy | A-Z2-9, no I/O/0/1 | **0/6** | 0.071 | 0/6 | 1/6 |
| hex | 0-9a-f | **0/6** | 0.048 | 0/6 | 3/6 |
| b64 | A-Za-z0-9+/ (I, l, O, 0 present) | **0/6** | 0.515 | 0/6 | 1/6 |

High-entropy decode across the base matrix: **0/18** exact.
A larger-cell extension (9x18, 10x20, 12x24, hex and base64 only) adds
1/12 - the single success was hex at 12x24.
Combined that is 1/30, but the
conditions differ and the pooled ratio should not be read as one probability. Per condition
n=2, and 0/2 supports only a one-sided 95% upper bound of 0.776.

**Bigger glyphs do not fix it.** Across 5x10, 6x13, 8x16, 9x18, 10x20 and 12x24,
mean hex CER moves 0.071 -> 0.018 while pixels per character rise 5.8x. Exact match
stays at 0/2 in every cell but one (12x24 hex, 1/2). The residual is a composite -

    D_total = D_visual + D_sequence + D_boundary + D_binding + D_abstention

with evidence for each: base64 `f`/`F` confusions (visual), long records failing where
short fields do not (sequence), one 51-character answer with 50 correct characters and
only the last wrong (boundary), returned text matching no row (binding), and outright
UNREADABLE responses (abstention). Optical resolution still matters - the f/F case
proves it - so 'the residual is not optical' would be too strong.

### Short spans, n=20 per condition (`span_probe.py`)

Lowercase hex and Crockford base32 at 6x13, payload rendered in 4-character groups,
asked for 'the next N characters' from an anchored start:

| alphabet | symbols | span 8 | span 16 | span 32 | span 51 |
|---|---|---|---|---|---|
| hex | 16 | 18/20 | 16/20 | 15/20 | 0/20 |
| crock32 | 32 | 9/20 | 7/20 | 4/20 | 0/20 |

One-sided 95% Clopper-Pearson lower bounds for hex: span 8 -> 0.717, span 16 -> 0.599,
span 32 -> 0.544. An earlier 3-rep run of the same condition returned 3/3 and was
reported as 'byte-exact through 32 characters'. At n=20 it is 15/20. That was small-sample
luck in the favourable direction - the reason every rate here carries a bound.

### Delimited, equal-information fields, n=20 (`delim_probe.py`)

The screening comparison above is confounded: 32 hex characters carry 128 bits, 32 base32
characters carry 160, and scoring was literal. This one holds information constant, puts
the field in visible brackets so nothing has to be counted, and scores decoded **value**
equality (case-folding, separator removal and Crockford ambiguity mappings are part of the
decoder, so they are not a grading loophole - but they must then be part of the protocol
specification).

| bits | alphabet | chars | literal | value | rate | 95% LB |
|---|---|---|---|---|---|---|
| 64 | hex | 16 | 20/20 | **20/20** | 1.00 | 0.861 |
| 64 | b32 | 13 | 14/20 | 19/20 | 0.95 | 0.784 |
| 128 | hex | 32 | 16/20 | 16/20 | 0.80 | 0.599 |
| 128 | b32 | 26 | 12/20 | 15/20 | 0.75 | 0.544 |

**No encoding advantage was detected - which is not the same as equivalence.** The
difference in value-correct rate, with 95% intervals:

| bits | hex - base32 | 95% CI | McNemar (discordant pairs) |
|---|---|---|---|
| 64 | +0.05 | [-0.05, +0.15] | 1 vs 0, p = 1.000 |
| 128 | +0.05 | [-0.21, +0.31] | 3 vs 2, p = 1.000 |

Those intervals are wide enough to contain a materially better hex *and* a modestly
better base32. Establishing equivalence would need a pre-declared margin (say d = 0.10)
and an interval falling entirely inside [-d, +d]; n=20 cannot do that. The earlier wording
here, 'statistically indistinguishable', overstated a null result and is corrected.

**Instrument defect, now fixed.** The first version of this probe put the encoding name in
the RNG seed, so hex and base32 encoded *different* random values - the comparison was
unpaired, and an earlier version of this README wrongly claimed the same values were used.
The seed no longer includes the encoding, so both alphabets encode identical bitstrings and
McNemar applies. The unpaired run is kept as `results_delim_UNPAIRED_6x13_opus.json`.

**Between-run variation is large at n=20.** The same 128-bit hex condition scored 11/20 in
the unpaired run and 16/20 in the paired one - a 0.25 swing in the point estimate (Fisher
p = 0.19 between the two runs, so not a detected difference, but a reminder that a single
n=20 cell is not a precise measurement). The 64-bit base32 cell moved 17/20 to 19/20 the
same way. Only the 64-bit hex cell was stable at 20/20 across both.

Pairing also shows the two encodings mostly succeed and fail on the *same underlying*
values: only 1 discordant pair at 64 bits and 5 at 128. Item difficulty appears driven by
the value and its rendering rather than by the alphabet.

### Information-normalised throughput

Raw success rate is the wrong objective: base32 carries the same information in fewer
characters, so the comparison should be successfully decoded **bits** per image token,
`rho_chars x (B/L) x P_value` at `rho_chars = 10.1`:

| payload | chars | P(value) | successful bits/image-token |
|---|---|---|---|
| 64-bit hex | 16 | 1.00 | **40.4** |
| 64-bit b32 | 13 | 0.95 | **47.2** |
| 128-bit hex | 32 | 0.80 | **32.3** |
| 128-bit b32 | 26 | 0.75 | **37.3** |

On point estimates base32's shorter representation offsets its lower first-pass rate, so
the alphabet question is open rather than settled in hex's favour. These figures exclude
delimiters, record separators, page labels, checksums, retries and binding failures.

### The best observed field is a candidate, not a guarantee

A visibly delimited 16-character hex field (64 bits) at 6x13 returned 20/20 value-correct.
Pointwise that is a one-sided 95% lower bound of 0.861 - but it is the best of four
delimited conditions, selected after a long adaptive exploration of cells, layouts,
alphabets and span lengths. A four-cell Bonferroni adjustment alone would drop the bound to
about 0.803, and the wider selection history is not repairable by any adjustment. The
defensible statement is **best observed candidate**, pending one fresh held-out run of the
now-frozen design. On successfully decoded bits per image token the 13-character base32
field actually leads it (47.2 against 40.4), so 'best' depends on which objective is being
optimised - exact-field rate or information throughput.

It is also a *raw* 64-bit value, not a checksummed handle. A handle must spend some of
those bits on validation - for example a 32-bit record index plus a 32-bit keyed tag inside
the same 16-character span - and a structured codeword has a different symbol distribution
from a uniform random one, so it needs measuring in its own right.

### Grouping and delimiting interact with length

| field | grouped payload, counted span | contiguous payload, delimited |
|---|---|---|
| 16 chars | 16/20 | **20/20** |
| 32 chars | 15/20 | **11/20** |

Removing the counting burden helped at 16 characters and hurt at 32, so 15/20 was not a
'floor' for delimited designs. Two variables changed together (grouping vs contiguity, and
counted vs delimited), so no separate effect is identified - an earlier claim here that
visible grouping stabilises the per-symbol rate is unsupported and withdrawn. The opposite
directions at 16 and 32 characters also rule out one constant per-character success
probability across layouts. The minimum causal experiment is a paired 2x2:

    {grouped, contiguous} x {counted, delimited},  at 16 and 32 chars,
    identical underlying values and identical image positions

which separates grouping benefit, delimiter benefit, counting cost, and their interaction
with length.


---

## 4. The channel is sparse symbol substitution

Fitting `P_exact(n) = q^n` jointly across spans 8/16/32 for grouped lowercase hex at 6x13:

    q = 0.9890      per-symbol error 1.10%

| span | observed | q^n predicted |
|---|---|---|
| 8 | 18/20 | 0.915 |
| 16 | 16/20 | 0.838 |
| 32 | 15/20 | 0.702 |
| 51 | **0/20** | 0.569 |

Spans 8-32 fit one per-symbol rate closely. Span 51 does not: under the fitted model
`P(0/20) = 4.9e-08`. The 51-character task is a **separate failure regime** - counting,
boundary tracking, output-span control or abstention - not more of the same optical noise.
(An earlier reading of the 3-rep data claimed the spans refuted a per-character model; with
n=20 the model fits, and that claim is withdrawn.)

Failures are also sparse: **1.0 character edits per failed record** at spans 16 and 32, 2.0
at span 8. Nearly every failure is a single wrong symbol.

**So the engineering response is detection and correction, not pixels.** Larger cells did
not move CER; a single-symbol-correcting code over a short field would recover almost every
observed failure. The metric that then matters is not `P(literal mismatch)` but

    P(a corrupted handle resolves to a DIFFERENT valid record)

driven to zero by checksum plus unique-match rejection - never by trusting the transcription.

---

## 5. Binding is the dominant failure

Decoding correctly says nothing about *provenance*. The bind probe asked for a named row
and scored which row actually came back:

| payload | bind exact | returned text matched no row at all |
|---|---|---|
| prose | 3/6 | 2/6 |
| alnum | 0/6 | 5/6 |
| hex | 0/6 | 3/6 |
| b64 | 0/6 | 5/6 |

In the exploratory ladder the same effect appeared as legibility 1.00 with lookup 0%: the
model transcribed an adjacent row perfectly. That is a referential error, not an OCR error,
and it is more dangerous because the returned string is well-formed.

Two limits on reading this. 0/6 per condition gives a one-sided 95% upper bound of 0.393,
so 'near zero' overstates it; and pooling heterogeneous payloads to n=18 is invalid for
estimating one probability. What is established is that **dense row-number addressing is a
bad protocol**, not that binding is an intrinsic optical limit.

Page shape does not rescue it. At a fixed cell, 74-row, 112-row and 149-row pages all gave
legibility 0.93-0.97 and lookup accuracy 0-25%. Fewer rows did not help.

Hierarchical addressing - page ID, block ID every 8-16 records, short record handle, with a
textual manifest outside the bitmap - is **untested** and is the largest open item here.

### A checksum does not fix this

Two failure classes look identical from outside and only one is detectable:

| class | what happens | detectable by checksum |
|---|---|---|
| lexical corruption | right record selected, handle misread: `H_i -> H_hat != H_i` | **yes** |
| provenance corruption | wrong record selected, its *valid* handle read correctly: `R_j -> H_j` | **no** |

The second passes validation because `H_j` is a well-formed codeword for a real record. So a
validation tag solves transcription failure and does nothing for binding failure, which is
the one the bind probe actually found. Any production path needs semantic or structural
reconciliation after the canonical fetch - confirming the retrieved record is the object that
was asked for - not just a passing checksum.

---

## 6. Density taxonomy

| quantity | 6x13 value | meaning |
|---|---|---|
| `rho_gross` | 10.1 | characters physically rendered per image token |
| `rho_symbol` | ~10.0 | expected *correct* symbols per image token (CER 0.008) |
| `rho_anchored` | ~7.6 | all-or-nothing record yield at the 32-char success rate |
| `rho_bits` | ~40-42 | successfully decoded *bits* per image token, short fields |
| `rho_operational` | **unknown** | includes binding, checksum rejection, retries, fetch |

The end-to-end quantity is bits of *canonical* information resolved per visual token:

    rho_resolved = rho_bits x P(correct record bound)
                            x P(validation accepts only when it should)
                            x P(canonical fetch returns the right record)

Only the first factor is measured. `P(false accept)` - a corrupted code validating as a
different real record - is not measured at all, and is the number that decides whether the
channel is safe rather than merely dense.

`rho_anchored` assumes one correctly anchored record, zero utility for partial reads, and no
cost for retries or canonical fetch. It is a throughput proxy, not a channel capacity. An
earlier version of this file called it `rho_operational`; the span probe supplied the anchor,
so it never measured `P(correct anchor)` at all.

---

## 7. What limits density (exploratory)

> Everything in this section is **exploratory**. The prompt, tool policy, grader, payloads,
> line lengths and candidate fonts were changed during the run in response to results, and
> the reporting threshold was chosen after seeing scores. Treat it as hypothesis generation.
> Sections 1-5 are the measured claims.

**Legibility is not one variable.** Separating glyph resolution from row addressing, and
redundant text from unguessable text, splits the apparent 'floor' into three different limits:

* **Row pitch** matters more than glyph area. 6x9 (54 px/char) reads where clR8x8 (64 px/char,
  larger glyphs) fails - everything at 8px pitch or below failed, everything at 9px and above
  read. But pitch alone is not sufficient: 4px-wide glyphs failed at pitch 9 and 10 and only
  recovered at pitch 12, so width and pitch interact.
* **Payload redundancy** is worth more than any font choice. At 6x10 - a cell that reads
  template prose at ~0.95 - the redundancy credit against unguessable payload reached +0.94.
  A closed-template corpus lets a model pattern-match rather than resolve glyphs.
* **Line length** confounds legibility with output stamina. The same cell scored 0.59 on
  224-character lines and 0.95 on 56-character lines.

The ladders themselves (bitmap 4x6 to 12x24, TrueType 5-32px, proportional prose) live in
`images/A_*`, `B_*`, `C_*` with per-image density in `manifest.csv`.

---

## 8. Method

The confirmatory harness freezes everything before running: prompts (sha
7ec0b7be25263cad / 3610002e12ac3ba4), harness (sha 89678907c6e937d8),
grader, tool policy, line length (56 chars), and a fresh seed namespace.
Raw responses are retained. CLI 2.1.234 (Claude Code).

* **Tools are locked to `Read`.** Without that, a model will shell out to Python to crop and
  zoom the image, which measures tooling rather than vision. That happened, and is why
  `Bash,Write,Edit,Glob,Grep,Task,WebFetch,WebSearch` are explicitly disallowed.
* **Abstention is distinguishable from error.** Prompts ask for `UNREADABLE` rather than a
  guess, and the corpus is random enough that a guess is always wrong.
* **Grading separates decoding from addressing.** For any wrong answer the grader searches
  every ground-truth line for what the model actually returned, so reading the right
  characters off the wrong row is scored as a binding error, not a legibility one.
* **Value vs literal equality.** Hex and base32 tolerate case-folding and regrouping without
  changing the decoded value, so both are reported. For base64 case is semantic and no such
  normalisation is permitted.
* **n is small.** Two repetitions per cell/payload in the confirmatory matrix, 20 in the
  screening and delimited stages. 59/59 clean runs would be needed to put a one-sided 95%
  bound above 0.95, and 299/299 above 0.99. Nothing here reaches that.

---

## 9. Confounds found, including in these instruments

Most of the useful findings here came from discovering that a measurement was measuring
something else.

| confound | effect | fix |
|---|---|---|
| Rater shared context with the corpus generator | reading becomes recall with a visual hint | fresh `claude -p` process per run |
| Closed-template corpus | inflates legibility up to +0.94 vs unguessable payload | series K/K2 high-entropy payloads |
| Confusable-free alphabet (no I/O/0/1) | optimistic for real hashes and hex | hex / base64 / base32 added |
| 224-character lines | measures output stamina, not reading | line length held at 56 |
| 'transcribe the next N characters' | measures counting too | delimited fields |
| Equal character count across alphabets | 160 bits vs 128 - not a fair comparison | equal-information encoding |
| Literal string scoring | penalises value-preserving case/grouping | value equality reported alongside |
| Model given non-Read tools | crops and zooms instead of reading | tools disallowed |
| Adaptive test selection, post-hoc threshold | winner's curse | frozen confirmatory pass |

Three harness bugs also corrupted results before being caught: Pillow mis-decodes X11 PCF
fonts declaring `first_col > 0` (8x16 and 12x24 rendered one glyph shifted - patched in
`gen_lib.py`); `edge_probe.py` overwrote its results file each run; and stored answers were
truncated to 120 characters before being compared against full-length ground truth.

---

## 10. What was retracted

| claim | status | replaced by |
|---|---|---|
| 1568x1568 is the maximum canvas | **retracted** | 1932x1932, measured at the adjacent boundary |
| Canvas must be a multiple of 224 | **narrowed** | `lcm(28, cw)` / `lcm(28, ch)` for Claude alone |
| Claude downscales above ~1.15 MP | **retracted** | no downscaling to 3.63 MP on this path |
| Row pitch >= 9 is the rule | **narrowed** | necessary, not sufficient - 4px glyphs fail at pitch 9-10 |
| 0.95-0.98 similarity means near-exact | **retracted** | exact match 0/18 at those scores |
| P(exact) = 0 for high-entropy payload | **narrowed** | no tested config demonstrated it; 0/18 gives UB 0.153 |
| Bigger glyphs do not fix exactness | **narrowed** | composite failure; optical resolution still matters |
| 9/9 exact through 32 characters | **retracted** | 15/20 at n=20 - small-sample luck |
| The span data refutes a per-character model | **retracted** | q = 0.989 fits spans 8-32 |
| Alphabet size dominates confusion-resistant design | **retracted** | indistinguishable at equal information |
| 15/20 is a floor for delimited fields | **withdrawn** | delimited scored worse at 32 chars |
| rho_operational ~= 10 chars/token | **renamed** | `rho_anchored`; operational still unknown |
| 8x16 is safe for anything | **retracted** | tested ASCII only, and 0/6 exact on whole records |

---

## 11. Files

| script | what it does |
|---|---|
| `gen_lib.py` | fonts, token maths, corpus generators, PCF fix |
| `generate.py` | the 92-image exploratory corpus (series A-G) |
| `pack.py` | zero-waste canvas solver; series H-K2 |
| `pack_text.py` | packs arbitrary text into the fewest image tokens |
| `run_eval.py` | ladder eval against `claude -p`, addressing-aware grader |
| `edge_probe.py` | legibility isolated from row addressing |
| `confirm.py` | frozen confirmatory pass, exact match + binding displacement |
| `span_probe.py` | exact match vs requested span length |
| `delim_probe.py` | delimited equal-information fields, value equality |
| `regrade.py` | re-scores stored answers without new API calls |
| `probe.py`, `report_eval.py`, `make_docs.py` | inspection and reporting |

```bash
python3 generate.py && python3 pack.py      # build images
python3 confirm.py --model opus --reps 2    # frozen pass
python3 make_docs.py                        # rebuild this file
```

Content is seeded per image, so a rebuild reproduces byte-identical images and answers.

---

## 12. Open questions

1. **Hierarchical binding.** Page -> block -> handle addressing with a textual manifest,
   reporting `P(page)`, `P(block|page)`, `P(record|block)` separately. The current failure is
   a property of row-number lookup, and nothing better has been measured.
2. **The admissible canvas region.** One rectangle was tested and failed. The long-edge limit,
   and whether moderate rectangles improve row count or cell divisibility, are unknown.
3. **Delimited x grouped, factorially.** The two were changed together; their separate
   contributions are unmeasured.
4. **Short fields at larger cells.** The n=20 screen ran only at 6x13. 9x18 and 12x24 may
   support longer exact fields; the whole-record CER result does not settle it.
5. **Error-corrected handles.** Test a structured codeword - 32-bit record index plus 32-bit
   keyed validation tag in the same 16-character span - and measure `P(false accept)`
   separately from first-pass failure. A structured codeword's symbol distribution differs
   from a uniform random one, so the 20/20 result does not transfer to it untested.
6. **Equivalence, not just non-significance.** Declare a margin (d = 0.10) and power the
   hex/base32 comparison to fit the difference interval inside it. n=20 cannot.
7. **The 2x2 layout experiment.** Grouping x delimiting at 16 and 32 characters, paired.
8. **Other endpoints.** Every geometry probe ran on the Sonnet CLI path.
