# Dense text in images

How many characters fit in one image token, and how many of them come back correctly -
measured for **Claude** (28x28 px patches) and the **GPT-5.6 / Sol line** (32x32 px patches).
161 generated test images, 2,572 graded model runs, and a
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
| Resolves the right record best | **claude** (E-BIND-1 20/20 vs sol 17/20, luna 1/20) |
| Exact recovery of a 56-char record | Claude **0/18**; gpt-5.6 2-4/18 (underpowered, CIs overlap) |
| Biggest safety gap | sol returns a record for **~40%** of no-answer queries; Claude 0% |

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
| prose | closed-vocabulary English | **14/24** | 0.012 | 9/24 | 22/24 |
| alnum_easy | A-Z2-9, no I/O/0/1 | **1/24** | 0.211 | 0/24 | 9/24 |
| hex | 0-9a-f | **7/24** | 0.070 | 2/24 | 12/24 |
| b64 | A-Za-z0-9+/ (I, l, O, 0 present) | **1/24** | 0.243 | 0/24 | 12/24 |

High-entropy decode across the base matrix: **9/72** exact.
A larger-cell extension (9x18, 10x20, 12x24, hex and base64 only) adds
19/48 - the single success was hex at 12x24.
Combined that is 28/120, but the
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

**Decoder audit.** Of 34 value-correct base32 responses:

| form | count | status |
|---|---|---|
| canonical literal output | 26 | exact string match |
| required the permitted `O` -> `0` alias | 8 | valid *tolerant input*, not canonical output |
| required a forbidden non-canonical form, range truncation or excess-bit masking | **0** | none |

Those three rows are different things, and 'no non-canonical acceptances' would blur them.
What is established is that no acceptance depended on discarding excess bits - the second
row is the protocol working as specified, not the scorer over-accepting.

Every alias rescue was the same confusion: `O` read for the printed digit `0`, 8 of 8. No
case-only, no separator-only, no `I`/`L`. Since the Crockford alphabet excludes `O` from
ground truth entirely, the model emits a symbol that cannot occur there.

**Caveat on the decoder itself.** `dec_b32` returns the full integer; it does not reject
over-range values internally. Here the *scorer* rejects them because full-value comparison
against a known expected value fails. Production cannot rely on that, since it starts from
an unknown decoded handle, so a real validator must enforce `0 <= v < 2^B` **and**
`encode(v) == normalize(s)` before the tag is checked.

| bits | alphabet | chars | literal | value | rate | 95% LB |
|---|---|---|---|---|---|---|
| 64 | hex | 16 | 20/20 | **20/20** | 1.00 | 0.861 |
| 64 | b32 | 13 | 14/20 | 19/20 | 0.95 | 0.784 |
| 128 | hex | 32 | 16/20 | 16/20 | 0.80 | 0.599 |
| 128 | b32 | 26 | 12/20 | 15/20 | 0.75 | 0.544 |

**No encoding advantage was detected - which is not the same as equivalence.** The
difference in value-correct rate, with 95% intervals:

| bits | hex - base32 | 95% CI (matched-pair) | McNemar (discordant pairs) |
|---|---|---|---|
| 64 | +0.05 | [-0.05, +0.15] | 1 vs 0, p = 1.000 |
| 128 | +0.05 | [-0.17, +0.27] | 3 vs 2, p = 1.000 |

Intervals are matched-pair, not independent-sample: the same bitstrings are encoded
both ways, and within-pair association narrows the variance. With 1 and 5 discordant
pairs, McNemar has almost no power - `p = 1.000` means the discordant evidence is too
sparse to separate the encodings, not that they are equal. A matched-pair *exact* or
score-based interval would be preferable to this Wald approximation at these counts.

Those intervals are wide enough to contain a materially better hex *and* a modestly
better base32. Establishing equivalence would need a pre-declared margin (say d = 0.10)
and an interval falling entirely inside [-d, +d]; n=20 cannot do that. The earlier wording
here, 'statistically indistinguishable', overstated a null result and is corrected.

**Instrument defect, now fixed.** The first version of this probe put the encoding name in
the RNG seed, so hex and base32 encoded *different* random values - the comparison was
unpaired, and an earlier version of this README wrongly claimed the same values were used.
The seed no longer includes the encoding, so both alphabets encode identical bitstrings and
McNemar applies. The unpaired run is kept as `results_delim_UNPAIRED_6x13_opus.json`.

**Between-run swings are large, but confounded.** The 128-bit hex condition scored 11/20 in
the unpaired run and 16/20 in the paired one. Because the seed fix also changed the random
values, that 0.25 swing mixes item-set variation with model and provider stochasticity - it
is *between-dataset* variation under the same nominal condition, not an estimate of
repeated-call variance. Isolating the latter needs the identical 20 frozen images rerun
several times. The 64-bit hex cell scored 20/20 in both, i.e. 40/40 across two distinct item
sets (unadjusted one-sided 95% LB 0.928). That is **replication across item sets within the
same pipeline**, not independent confirmation: both runs share model family, prompt,
renderer, endpoint, field layout, grader and selection history. It justifies freezing the
format as the prototype candidate, nothing stronger.

Pairing shows elevated concordance, but less than it first appears. At 128 bits the two
encodings agree on 0.75 of pairs against 0.65 expected under independence - an excess of only 0.10, phi = 0.29.
That is suggestive of a shared per-item or per-call difficulty component, **not** an
identified value effect: shared outcomes could equally come from target position,
surrounding page composition, or call-level variation. At 64 bits hex has no failures, so
no association can be estimated at all. Separating them needs each bitstring repeated
across positions, pages and calls in both encodings, with value / position / call as
random effects.

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
the alphabet question is open rather than settled in hex's favour.

**This is anchored *payload* goodput, not page-level goodput.** It assumes the packer
actually reclaims the three saved cells. If handles sit in a fixed 16-character slot for
visual alignment then `L_physical = 16` for both encodings and base32's advantage mostly
disappears. It also excludes delimiters, separators, page and block labels, tag bits,
unused cells, retries and binding failure. The decisive comparison must be computed from
complete rendered protocol pages:

    G_page = sum_i B_i x 1[record i correctly resolved] / V(complete page)

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

It is also a *raw* 64-bit value, not a checksummed handle. A handle must spend those bits
as a budget across record identity, context binding and error detection:

For a t-bit keyed tag, `P_FA <= M / 2^t` where **M is the number of *distinct* wrong
candidates validated** - not the number of model calls, since repeatedly producing the same
bad candidate is not an independent trial. The bound assumes a truncated secure keyed MAC
or PRF, a key untouchable by untrusted content, canonicalisation before verification, and a
tag binding all relevant context.

| tag width | false accepts over 1e6 distinct bad candidates | remaining bits for identity |
|---|---|---|
| 32-bit | ~2.3e-4 | 32 |
| 40-bit | ~9.1e-7 | 24 |
| 48-bit | ~3.6e-9 | 16 |

A hierarchical protocol cuts the identity requirement: with page and block already
established, a 16-bit *local* record index plus a 48-bit keyed tag still fits the measured
64-bit span. The tag should cover archive, page, block, record index and a record
fingerprint, so a handle lifted from the wrong page fails validation **when the expected
context is independently known**. It cannot catch a correctly-read handle for the wrong
record inside an accepted context - only reconciliation after fetch does that.

A structured codeword also has a different symbol distribution from a uniform random value,
so the 20/20 result does not transfer to it untested.

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
87a08dbd92ac5be4 / bc791e06d6f9d11e), harness (sha ff3acf98bea39cf5),
grader, tool policy, line length (56 chars), and a fresh seed namespace.
Raw responses are retained. CLI codex-cli 0.147.0.

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

Six harness bugs corrupted results before being caught. From the Claude arm: Pillow
mis-decodes X11 PCF fonts declaring `first_col > 0` (8x16 and 12x24 rendered one glyph
shifted - patched in `gen_lib.py`); `edge_probe.py` overwrote its results file each run; and
stored answers were truncated to 120 characters before comparison against full-length ground
truth. From the GPT-5.6 arm:

| bug | symptom | fix |
|---|---|---|
| Prompt caching hid image tokens | a repeat call puts the image in `cached_input_tokens`, so `input + cache_creation` collapsed to ~0 and the first Opus geometry probe returned garbage | render **unique content for every measurement call** so the image can never be a cache hit |
| `codex exec -i` is variadic | a positional prompt placed after `-i` is swallowed as another image file; the call failed with 'No prompt provided via stdin' | pass the prompt on **stdin**, which also avoids argv limits on long prompts |
| `tokens()` omitted the 1.2 multiplier | GPT-5.6 densities in the answer key were inflated by exactly 1.2x, putting some images *above* their theoretical ceiling | recomputed every token field from geometry; **156 of 161 entries were wrong** |

The third is the most instructive: it was caught only because a sanity check asserted that no
image may exceed `1024/1.2/(cw*ch)` chars per token. Without that assertion the inflated
figures would have looked plausible and propagated into every Part II conclusion.

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
| *Part II* | | |
| A 32x32 patch costs 1 token | **retracted** | GPT-5.6 bills 1.2 tokens/patch; 853 px/token, not 1024 |
| Claude's 1932px ceiling is a general limit | **retracted** | GPT-5.6 caps at 1600x1600 / 2,500 patches |
| The span-51 cliff is a property of vision models | **retracted** | Claude-specific; all three 5.6 models read it |
| Enlarging glyphs cannot fix exactness | **narrowed to Claude** | terra reaches 8/12 on large cells |
| 'GPT-5.6 reads dense text better' | **retracted as a family claim** | only sol; luna and terra are significantly worse |
| Sol's abstention failure is family-wide | **retracted** | sol only; terra and luna abstain correctly |
| Better legibility implies better record resolution | **retracted** | sol reads best, Claude resolves best |

---

## 11. Files

| script | what it does |
|---|---|
| `providers.py` | one call interface over `claude -p` and `codex exec`; usage extraction |
| `geometry.py` | provider-agnostic px/token and downscale-ceiling probe |
| `gen_lib.py` | fonts, token maths, `PROVIDER_GEOMETRY`, corpus generators, PCF fix |
| `generate.py` | the 92-image exploratory corpus (series A-G) |
| `pack.py` | zero-waste canvas solver; series H-K2 |
| `pack_text.py` | packs arbitrary text into the fewest image tokens |
| `run_eval.py` | ladder eval, addressing-aware grader |
| `edge_probe.py` | legibility isolated from row addressing |
| `confirm.py` | frozen confirmatory pass, exact match + binding displacement |
| `span_probe.py` | exact match vs requested span length |
| `delim_probe.py` | delimited equal-information fields, value equality |
| `ebind1.py` | full protocol test: keyed codewords, canonical fetch, gold scoring |
| `regrade.py` | re-scores stored answers without new API calls |
| `compare.py` | cross-provider tables on paired, byte-identical stimuli |
| `probe.py`, `report_eval.py`, `make_docs.py` | inspection and reporting |

```bash
python3 generate.py && python3 pack.py      # build images
python3 geometry.py --model gpt-5.6-sol --patch 32   # px/token + ceiling
python3 confirm.py --model gpt-5.6-sol --effort low  # frozen pass, any provider
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

---

## 13. E-BIND-1: design

> **Status: executed.** This section is the design and its rationale; measured results for
> both providers are in section 23. Kept separate because the design decisions were frozen
> before any data was collected, and that ordering is the point.

Stop sweeping fonts, alphabets and isolated field accuracy. The remaining uncertainty is
whether a semantic query can resolve the *correct canonical record*. Implemented in
`ebind1.py`.

### Pin one model - do not compose across them

Every fidelity result in this file was measured on **Opus**; every *original* geometry result,
including the 1932x1932 ceiling and the 784 px/token rate, was measured on **Sonnet**. That
composition was unverified, and preprocessing, downscaling and effort behaviour can all differ
by model. It has since been checked directly: the adjacent-boundary probe was repeated on both
models and both cap at 4,761 patches with the same 1932 boundary (section 1), so the geometry
transfers. The *fidelity* numbers remain Opus-only. E-BIND-1 still pins one exact model id,
endpoint and CLI version for both halves, and mutable aliases (`opus`, `sonnet`) must not
appear in the final provenance record.

### Gold scorer is not a runtime verifier

The evaluator knows which record generated each query, so it scores `R_fetched == R_gold`
deterministically. That is the **primary** metric. A deployable system does not know
`R_gold`, so any runtime semantic verifier is a separate component and must be measured as a
second arm - reporting `P(verifier accepts | R_fetched != R_gold)`. Implementing the
'semantic verifier' with the hidden answer key would make false acceptance impossible by
construction and would describe nothing deployable.

### Two outcome partitions, because abstention flips sign

| answer-present | | no-answer | |
|---|---|---|---|
| `C` | correct gold record accepted | `N` | correct NO_MATCH |
| `D` | wrong/corrupt candidate rejected | `W0` | **any record accepted** |
| `W` | **wrong canonical record accepted** | `D0` | invalid candidate rejected |
| `A` | abstained | `P0` | parse/protocol failure |
| `P` | parse/protocol failure | | |

The two load-bearing rates are `P(W | answer present)` and `P(W0 | no answer)`. Abstention is
correct behaviour in one partition and failure in the other, so they are never pooled.

### Two arms, because one layout cannot do both jobs

* **Arm A - fixed slot.** Both encodings occupy the same physical 16-character slot, so record
  positions, neighbours, line lengths and page geometry are identical. Base32's padding is
  deliberately wasted. This isolates encoding accuracy.
* **Arm B - native density.** Hex uses 16 cells, base32 13, pages packed independently. This
  measures `G_page` and is an architectural comparison, not a paired glyph-level one.

Asking one layout to do both would either shift the page (breaking pairing) or pad base32
(erasing the density advantage it is supposed to demonstrate).

### Structural IDs outside the bitmap - and the residual binding channel

    ARCHIVE A1 - PAGE P07 - BLOCK B03: <image path>      <- ordinary exact text
    [dense image payload]                                <- semantic content + record codewords

Page and block identifiers are tiny; compressing them optically adds binding uncertainty for
no density gain. But an exact *text* label can still be associated with the wrong adjacent
image, so that channel is measured rather than assumed. The keyed tag binds page and block, so
`P(tag pass | code read exactly)` reports label-image consistency for free: a code lifted from
the wrong block fails validation.

### Codeword

8-bit local index + 56-bit keyed tag over
`protocol || archive || revision || page || block || index || digest(record)`. Blocks hold
8-16 records so 8 index bits suffice, and the archive revision binds a handle to a canonical
state generation - a stale handle cannot silently resolve against changed state. Tag width is
not the bottleneck: the dominant error is selecting another real record carrying its own valid
tag, which no tag width addresses.

### Sampling - repeated calls are not extra items

100 unique answer-present queries + 20 no-answer, paired across encodings, x3 passes over the
frozen set = 720 calls. That is **120 independent semantic items**, not 720. The passes
estimate call variance conditional on those items; they do not multiply the item sample.
Analysis is repeated-measures:

    logit P(Y_ier = 1) = b0 + b_enc + u_item + v_page + w_pass

with a hierarchical bootstrap over items and pages as an alternative. For generalisation to
new items, `0/100` bounds the false-accept rate below only ~3%; the `0/299` -> ~1% and
`0/2995` -> ~0.1% figures require that many **independently constructed adversarial binding
opportunities**, not repeated calls over the same ones.

### Required controls

* **Wrong-valid-handle decoys**: semantically similar payload, different valid codeword, valid
  tag, same block where possible. A checksum rejects lexical corruption and accepts these;
  only post-fetch reconciliation rejects them. Without decoys a low false-accept count may
  only mean the test never produced a plausible wrong candidate.
* **Raw-text semantic ceiling**: identical queries, records, block structure, distractors and
  no-answer cases, carried as ordinary text. Without it a 65% optical score cannot be split
  into 95% semantic x 68% optical or 68% semantic x ~100% optical. One frozen pass suffices.
* **Preflight (Stage 0)**: ~20 queries per encoding to validate range checking, canonical
  re-encoding, tag construction, wrong-page/wrong-block rejection, same-block decoys reaching
  the semantic scorer, the outcome partition, raw retention and deterministic regrading - then
  freeze commit, renderer, prompts, model id and seed namespace, and regenerate a fresh
  held-out corpus. Preflight results are debug-only and must not enter the estimate.

### Stage 0 preflight result (debug-only, not an estimate)

75 calls on Opus, Arm A, 6 blocks x 12 records, 20 answer-present + 5 no-answer queries:

| carrier | C | D | W | N | W0 | P(correct block) |
|---|---|---|---|---|---|---|
| image, hex | **20/20** | 0 | **0** | 5/5 | **0** | 20/20 |
| image, base32 | 17/20 | 3 | **0** | 5/5 | **0** | 17/20 |
| text ceiling, hex | 20/20 | 0 | 0 | 5/5 | 0 | 20/20 |

The semantic ceiling is 20/20, so the query set is unambiguous and the optical arms are not
limited by semantic selection - hex matched the ceiling exactly. Compare this with row-number
addressing (section 5), which scored 0/6 on the same kind of task: **exact text labels plus a
short delimited keyed codeword is a working protocol where dense row lookup was not.**

The three base32 losses were 2 tag failures and 1 length failure (a 12-symbol code - a dropped
character), all clustered in one block. All three were *rejected*; none was accepted.

**What this does not show.** No wrong canonical record was ever accepted, but the model never
selected a wrong record, so the W path was never exercised by the model - exactly the
circumstance in which a zero false-accept count means nothing. The path was therefore verified
by direct injection instead:

| injected response | outcome | validator |
|---|---|---|
| a different real record in the same block | **W** | `ok` - the tag passes, as it must |
| the correct record | C | `ok` |
| any record on a no-answer query | **W0** | `ok` |
| a forged code | D | `tag_fail` |

So the partition is exhaustive and the dangerous case is detectable; it simply did not arise
at n=25 with 72 records. `0/20` bounds new-item false acceptance below only ~14%. Preflight
results are debug-only and do not enter any estimate.

### Reported metrics

`P(correct image block)`, `P(correct record | correct block)`, `P(correct block | text
ceiling)`, `P(canonical parse)`, `P(tag pass | correct record)`, `P(tag pass | corrupted
code)`, the full `C/D/W/A/P` and `N/W0/D0/P0` partitions, `G_page`, expected retries, p50/p95
latency and token cost. **`P(W)` is reported separately and never folded into a utility
score.**

---

## 14. Assumption register

| assumption | status | falsification probe |
|---|---|---|
| The 20 paired repetitions are independent | not explicitly established | confirm one fresh image and one independent call per rep |
| Pairing held target position and surrounding layout constant | unknown | compare renderer coordinates and distractor pages |
| No acceptance depended on excess-bit masking or truncation | **audited: yes** (0 of 34) | canonical re-encode audit |
| Tolerant normalisation is limited to protocol aliases | **audited: yes** (8 of 8 were `O`->`0`) | classify every literal-wrong/value-correct response |
| `dec_b32` itself rejects over-range values | **no** - the scorer catches them by full-value mismatch | add explicit range + re-encode validation before tag check |
| Matched-pair Wald intervals are accurate enough | approximate at 1 and 5 discordant pairs | exact or score-based matched-pair interval |
| 40/40 is independent confirmation | **too strong** - shared pipeline | treat as replication across item sets |
| base32 saves cells at page level | unknown under fixed slots and metadata | render complete protocol pages |
| Exact-text page/block labels preserve selection difficulty | plausible | compare against in-image labels |
| Semantic verification catches wrong valid records | unknown, load-bearing | include near-decoy records |
| Concordance reflects bitstring difficulty | suggestive only (phi = 0.29) | repeat each value across positions and calls |
| The two runs measure stochastic variation | **false as stated** - values differed too | rerun identical frozen images |
| base32 goodput advantage survives protocol overhead | plausible, unmeasured | include delimiters, manifests, validation, retries |
| A structured 64-bit handle behaves like a uniform value | unknown | test the actual codeword distribution |
| Hierarchical binding beats row addressing | unknown | run the page/block/record test |
| A 32x32 patch costs one token | **false** - GPT-5.6 bills 1.2 tokens/patch | fit tokens against patch count (section 15) |
| Claude's 1932px ceiling transfers to other providers | **false** - GPT-5.6 caps at 1600px | re-run `geometry.py` per provider |
| The 5.6 line beats Claude on high-entropy exact decode | directional only, n=18, CIs overlap | span/delim stages at n=20 per condition |
| The two providers are effort-matched | **false** - 5.6 pinned `low`, Claude uncontrolled | sweep effort on 5.6 |
| Image delivery is equivalent across providers | **false** - attached vs path+Read tool | unavoidable; state it on every table |
| The 5.6 line is uniform on this task | **false** - sol beats Claude on legibility, luna and terra are significantly worse | per-model paired tests (section 19) |
| Long-span reading failure is intrinsic to vision models | **false** - Claude 0/20 at span 51, all three 5.6 models 12-16/20 | span sweep on both providers |
| Bigger glyphs cannot fix exact recovery | **false for GPT-5.6** - terra 8/12 on large cells where opus manages 1/12 | confirm extension, both providers |
| Uppercase base32 is workable | **false for the entire 5.6 line** - opus 17/20, sol 4/20, terra 2/20, luna 0/20 | four independent probes |
| A model that reads better resolves records better | **false** - sol reads best, Claude resolves best | E-BIND-1 vs legibility (sections 19, 23) |
| Poor image results imply poor semantics | **false** - sol scores 99% on the text carrier and 63% on images | text-ceiling control |
| Abstention failure is a 5.6 family trait | **false** - only sol; terra and luna abstain 5/5 | preflight across three models |
| GPT-5.6 effort setting does not matter here | **unmeasured** - everything ran pinned at `low` | sweep low/medium/high on one condition |

---

# Part II - the GPT-5.6 (Sol) line

Run through `codex exec` against `gpt-5.6-sol`, `gpt-5.6-luna` and `gpt-5.6-terra`,
reasoning effort pinned `low`, on the **same images** the Claude arm used. 106 of the
128 images already have both dimensions divisible by 32 - a side effect of building the
corpus on multiples of 224 = lcm(28,32) - so both providers see byte-identical stimuli
with zero quantisation waste on either grid.

## 15. GPT-5.6 geometry

32x32 patches confirmed - **but a patch does not cost one token.** Fitting billed input
against patch count across 224/448/896/1344 squares:

    tokens = base + 1.2 x patches

| canvas | patches | sol tokens | implied base at 1.2 |
|---|---|---|---|
| 224x224 | 49 | 14,048 | 13,989.2 |
| 448x448 | 196 | 14,225 | 13,989.8 |
| 896x896 | 784 | 14,930 | 13,989.2 |
| 1344x1344 | 1,764 | 16,108 | 13,991.2 |

The base is constant to within 2 tokens across a 36x range in patch count, so the 1.2
multiplier is real, not noise. Effective rate: **1024/1.2 = 853 px per billed token**
against Claude's 784 - a uniform **+8.8% density advantage at every cell**.

### The ceiling is a patch cap at 1600x1600

1792, 2240, 2688 and 3200 pixel squares **all bill exactly 16,992**. Bisecting:

| canvas | patches | measured | uncapped prediction | verdict |
|---|---|---|---|---|
| 1536x1536 | 2,304 | 16,756 | 16,755 | uncapped |
| 1600x1600 | 2,500 | 16,992 | 16,990 | at the cap |
| 1632x1632 | 2,601 | 16,992 | 17,111 | **capped** |
| 1664x1664 | 2,704 | 16,992 | 17,235 | **capped** |

So anything past 1600px is downscaled back to 50x50 = 2,500 patches. Unlike Claude's
ceiling, this one needed no paired deltas or min-of-N: **GPT-5.6 token accounting is
deterministic**, returning identical values across reps at every size.

### Family-wide, not model-specific

| model | tokens/patch (896->1600) | base prompt | cap |
|---|---|---|---|
| gpt-5.6-sol | 1.2016 | 13,990 | 1600px |
| gpt-5.6-luna | 1.2016 | 12,426 | 1600px |
| gpt-5.6-terra | 1.2016 | 13,990 | 1600px |

Identical to four decimal places; only the base prompt length differs (system-prompt
size, not geometry). All three accept attached images.

## 16. Capacity: the two providers trade off in opposite directions

| | patch | tok/patch | px/token | max square | image tokens |
|---|---|---|---|---|---|
| claude | 28 | 1.0 | 784 | 1932x1932 | 4,761 |
| gpt-5.6 | 32 | 1.2 | **853** | **1600x1600** | 3,000 |

One maximum-size page:

| cell | claude chars | claude ch/tok | gpt-5.6 chars | gpt-5.6 ch/tok |
|---|---|---|---|---|
| 5x10 | **74,498** | 15.65 | 51,200 | **17.07** |
| 6x13 | **47,656** | 10.01 | 32,718 | **10.91** |
| 8x16 | **28,920** | 6.07 | 20,000 | **6.67** |

GPT-5.6 is denser **per token**; Claude holds more **per image**. Optimising cost per
character picks GPT-5.6; optimising characters per request picks Claude.

## 17. Fidelity on identical images

`confirm.py` base matrix - 5x10 / 6x13 / 8x16 x 2 reps, 56-character records:

| model | payload | decode exact | mean CER | returned the asked-for row |
|---|---|---|---|---|
| gpt-5.6-luna | prose | 3/6 | 0.024 | 3/6 |
| gpt-5.6-luna | alnum_easy | 0/6 | 0.372 | 2/6 |
| gpt-5.6-luna | hex | 2/6 | 0.027 | 2/6 |
| gpt-5.6-luna | b64 | 0/6 | 0.137 | 2/6 |
| gpt-5.6-sol | prose | 4/6 | 0.006 | 5/6 |
| gpt-5.6-sol | alnum_easy | 0/6 | 0.044 | 4/6 |
| gpt-5.6-sol | hex | 3/6 | 0.176 | 3/6 |
| gpt-5.6-sol | b64 | 0/6 | 0.095 | 5/6 |
| gpt-5.6-terra | prose | 2/6 | 0.015 | 6/6 |
| gpt-5.6-terra | alnum_easy | 1/6 | 0.357 | 1/6 |
| gpt-5.6-terra | hex | 2/6 | 0.030 | 3/6 |
| gpt-5.6-terra | b64 | 1/6 | 0.223 | 2/6 |

### Against Claude, on the metric that mattered most

High-entropy exact decode of a 56-character record - the number that was **0/18** on
Claude at every cell and alphabet:

| model | exact | rate | 95% CI | Fisher vs claude |
|---|---|---|---|---|
| opus (claude) | 0/18 | 0.00 | [0.00, 0.15] | - |
| gpt-5.6-luna | 2/18 | 0.11 | [0.02, 0.31] | 0.486 |
| gpt-5.6-sol | 3/18 | 0.17 | [0.05, 0.38] | 0.229 |
| gpt-5.6-terra | 4/18 | 0.22 | [0.08, 0.44] | 0.104 |
| 5.6 pooled | 9/54 | 0.17 | [0.09, 0.27] | 0.100 |

**No individual comparison is significant.** Every 5.6 model recovered records where
Claude recovered none, and the direction is consistent across all three, but the CIs
overlap and the pooled figure mixes 3 models x 3 cells x 3 payloads - the same
heterogeneous pooling criticised in section 5. The supportable claim is *a consistent
directional difference that n=18 cannot establish*, not a demonstrated advantage.

### Binding is where sol clearly separates

| model | returned the asked-for row | bind exact |
|---|---|---|
| opus (claude) | 6/24 | 3/24 |
| gpt-5.6-luna | 9/24 | 1/24 |
| **gpt-5.6-sol** | **17/24** | 6/24 |
| gpt-5.6-terra | 12/24 | 1/24 |

Row addressing was the dominant Claude failure and the reason E-BIND-1 routes around
it entirely. Sol hits the requested row 71% of the time against Claude's 25%, and
that gap separates sol from its own siblings as much as from Claude.

### Bigger glyphs DO fix exactness on GPT-5.6 - a Claude-specific finding, corrected

Section 3 concluded that enlarging glyphs does not eliminate exact-copy failures: Claude
managed 1/12 on whole 56-character records even at 9x18/10x20/12x24 (up to 288 px per
character). That is **not** a general property of vision models - it is a property of
Claude. On the same images:

| model | decode exact (larger cells) | mean CER |
|---|---|---|
| **gpt-5.6-terra** | **8/12** | 0.009 |
| gpt-5.6-sol | 6/12 | 0.098 |
| gpt-5.6-luna | 4/12 | 0.025 |
| opus | 1/12 | 0.046 |

So the remedy Part I ruled out - just render bigger - works on GPT-5.6 and not on
Claude. Note also that terra is the *worst* of the three at small-glyph legibility and
the *best* at exact transcription of large ones, so these are separable capabilities
and a single model ranking would hide that.

### Inversion worth keeping

On **prose** Claude was better (5/6 against sol's 4/6). The 5.6 advantage is specific
to unguessable payload - consistent with less reliance on language priors to repair
ambiguous glyphs, which is exactly the redundancy credit measured in section 3.

### Caveats attached to every number above

* **Image delivery differs.** `codex exec` attaches the image with `-i`; `claude -p`
  receives a path and fetches it with the Read tool. Stimuli identical, path not.
  This also means the shell-out-and-zoom confound is structurally impossible on
  codex, whereas Claude needed explicit tool disallowing.
* **Effort is pinned `low`** for the 5.6 models (their default). The Claude runs had
  no equivalent control, so the comparison is not effort-matched.
* **n = 18 per model** across mixed cells and payloads.


## 18. The bitmap ladder, same images as Claude

`run_eval.py` over series A. *lookup* = exact match on the code at a named line;
*legible* = similarity to whichever ground-truth line the answer best matches.

| glyph | ch/token (28-grid) | gpt-5.6-luna lookup | gpt-5.6-luna legible | gpt-5.6-sol lookup | gpt-5.6-sol legible | gpt-5.6-terra lookup | gpt-5.6-terra legible |
|---|---|---|---|---|---|---|---|
| 4x6 | 31.3 | 0% | 0.00 | 0% | 0.00 | 0% | 0.00 |
| 5x7 | 21.1 | 0% | 0.00 | 0% | 0.00 | 0% | 0.00 |
| 5x8 | 18.6 | 0% | 0.00 | 25% | 0.00 | 0% | 0.00 |
| 6x9 | 13.7 | 25% | 0.58 | 25% | 0.97 | 0% | 0.00 |
| 6x10 | 12.3 | 25% | 0.00 | 75% | 0.99 | 25% | 0.94 |
| 6x12 | 10.3 | 25% | 0.00 | 50% | 0.99 | 25% | 0.97 |
| 6x13 | 9.5 | 25% | 0.71 | 25% | 1.00 | 0% | 0.96 |
| 7x13 | 8.0 | 75% | 1.00 | 50% | 1.00 | 50% | 1.00 |
| 7x14 | 7.4 | 0% | 0.97 | 25% | 0.97 | 50% | 0.96 |
| 8x13 | 7.0 | 75% | 1.00 | 100% | 1.00 | 0% | 0.99 |
| 8x16 | 5.7 | 25% | 0.98 | 75% | 0.99 | 50% | 0.99 |
| 9x15 | 5.3 | 50% | 1.00 | 100% | 1.00 | 50% | 0.99 |
| 9x18 | 4.5 | 100% | 1.00 | 75% | 1.00 | 50% | 1.00 |
| 10x20 | 3.5 | 75% | 0.97 | 100% | 1.00 | 75% | 0.97 |
| 12x24 | 2.5 | 50% | 1.00 | 100% | 1.00 | 100% | 1.00 |

## 19. Legibility isolated from addressing

First and last lines only - they sit at the page edges, so no row counting is
involved and the score is glyph resolution alone. Byte-identical files across all
models, n=26.

| cell | ch/token | opus | gpt-5.6-luna | gpt-5.6-sol | gpt-5.6-terra |
|---|---|---|---|---|---|
| 4x9 | 21.5 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4x9 | 21.3 | 0.00 | 0.00 | 0.39 | 0.00 |
| 4x10 | 19.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5x9 | 17.2 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5x9 | 17.0 | 0.94 | 0.66 | 0.97 | 0.00 |
| 5x9 | 16.9 | 0.80 | 0.00 | 0.97 | 0.00 |
| 5x9 | 16.9 | 0.88 | 0.86 | 0.73 | 0.00 |
| 6x8 | 15.9 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4x12 | 15.9 | 0.90 | 0.00 | 0.96 | 0.00 |
| 6x8 | 15.6 | 0.00 | 0.74 | 0.73 | 0.70 |
| 5x10 | 15.2 | 0.99 | 0.67 | 0.98 | 0.70 |
| 5x10 | 15.2 | 0.99 | 0.19 | 0.98 | 0.00 |
| 6x9 | 14.2 | 0.00 | 0.00 | 0.55 | 0.00 |
| 6x9 | 14.1 | 0.84 | 0.58 | 0.99 | 0.96 |
| 6x9 | 14.0 | 0.55 | 0.54 | 0.90 | 0.17 |
| 6x10 | 12.7 | 0.00 | 0.18 | 0.93 | 0.54 |
| 6x10 | 12.6 | 0.70 | 0.20 | 0.96 | 0.66 |
| 6x13 | 9.8 | 0.59 | 0.66 | 0.93 | 0.00 |
| 6x13 | 9.7 | 0.95 | 0.56 | 0.94 | 0.87 |
| 7x14 | 8.0 | 0.69 | 0.93 | 0.97 | 0.77 |
| 8x16 | 5.9 | 0.98 | 0.94 | 0.98 | 0.97 |
| 8x16 | 5.9 | 0.96 | 0.92 | 0.99 | 0.86 |
| 9x18 | 4.7 | 1.00 | 0.97 | 0.98 | 0.97 |
| 9x18 | 4.6 | 1.00 | 0.98 | 0.98 | 0.98 |
| 10x20 | 3.8 | 0.99 | 0.94 | 0.98 | 0.95 |
| 12x24 | 2.6 | 1.00 | 0.96 | 0.99 | 0.97 |

| model | mean legibility |
|---|---|
| gpt-5.6-sol | **0.761** |
| opus | **0.606** |
| gpt-5.6-luna | **0.480** |
| gpt-5.6-terra | **0.426** |

Three bands. Below ~8 ch/token every model scores 0.94-1.00 and nothing is
distinguishable. Between 9.7 and 15.2 sol dominates while opus is erratic (0.00 to
0.99 across neighbouring conditions) and terra/luna are weak. Above 16.9 only sol
and opus read at all - and **sol alone reads anything at 21.3 ch/token** (0.39
where every other model scores 0.00).

## 20. Short spans at n=20 - two metrics, because they measure different things

`exact` requires the answer to be exactly the N characters requested. `prefix` asks
only whether the answer *starts* with them. The gap is not a reading difference: sol
frequently returns N+1 characters, completing the visible 4-character group rather than
cutting mid-group. Opus scores identically on both metrics (49/80 either way) - it
always cuts at exactly N - so the split isolates **instruction-following** from
**reading**.

| model | alphabet | metric | span 8 | span 16 | span 32 | span 51 | total |
|---|---|---|---|---|---|---|---|
| opus (claude) | hex | literal | 18/20 | 16/20 | 15/20 | 0/20 | **49/80** |
| gpt-5.6-luna | crock32 | literal | 7/20 | 4/20 | 0/20 | 0/20 | **11/80** |
| gpt-5.6-luna | crock32 | **prefix** | 7/20 | 7/20 | 1/20 | 0/20 | **15/80** |
| gpt-5.6-sol | crock32 | literal | 2/20 | 8/20 | 6/20 | 1/20 | **17/80** |
| gpt-5.6-sol | crock32 | **prefix** | 2/20 | 8/20 | 6/20 | 1/20 | **17/80** |
| gpt-5.6-terra | crock32 | literal | 10/20 | 5/20 | 2/20 | 0/20 | **17/80** |
| gpt-5.6-terra | crock32 | **prefix** | 11/20 | 5/20 | 2/20 | 0/20 | **18/80** |
| gpt-5.6-luna | hex | literal | 14/20 | 14/20 | 4/20 | 12/20 | **44/80** |
| gpt-5.6-luna | hex | **prefix** | 17/20 | 16/20 | 10/20 | 12/20 | **55/80** |
| gpt-5.6-sol | hex | literal | 8/20 | 14/20 | 17/20 | 16/20 | **55/80** |
| gpt-5.6-sol | hex | **prefix** | 18/20 | 19/20 | 20/20 | 16/20 | **73/80** |
| gpt-5.6-terra | hex | literal | 17/20 | 14/20 | 12/20 | 14/20 | **57/80** |
| gpt-5.6-terra | hex | **prefix** | 18/20 | 14/20 | 17/20 | 14/20 | **63/80** |

**Alphabet dominates model.** Crockford base32 fails for everyone at 6x13 -
opus 20/80, sol 17/80, luna 15/80, no meaningful separation - while hex separates the
models cleanly (sol 73/80, luna 55/80, opus 49/80). Whatever makes uppercase base32
hard at this size defeats every model tested, so alphabet choice is a bigger lever
than model choice.

**Do not read hex vs crock32 here as an alphabet comparison.** 32 crock32 symbols
carry 160 bits against hex's 128, so the crock32 task is simply harder - the same
equal-character-count confound identified in section 3. The delimited probe (section 21)
holds *information* constant and is the clean test. What this table does show is that
crock32 prefix-tolerance rescues nothing (17 -> 17) whereas hex gains 18, so crock32
failures are genuine misreads rather than boundary artifacts.

Two results stand out. At **span 51 Claude scores 0/20** - a hard failure regime I
attributed to counting, boundary tracking or output stamina (section 4) - while sol
reads it 16/20. And at **span 32 sol is 20/20 prefix-correct**: a 128-bit handle read
byte-exact, twenty times out of twenty, where Claude managed 15/20.

## 21. Delimited equal-information fields at n=20

| model | bits | alphabet | chars | literal | value | 95% LB |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | 64 | hex | 16 | 10/20 | **10/20** | 0.302 |
| gpt-5.6-luna | 64 | b32 | 13 | 4/20 | **4/20** | 0.071 |
| gpt-5.6-luna | 128 | hex | 32 | 7/20 | **7/20** | 0.177 |
| gpt-5.6-luna | 128 | b32 | 26 | 1/20 | **1/20** | 0.003 |
| gpt-5.6-sol | 64 | hex | 16 | 17/20 | **17/20** | 0.656 |
| gpt-5.6-sol | 64 | b32 | 13 | 15/20 | **16/20** | 0.599 |
| gpt-5.6-sol | 128 | hex | 32 | 18/20 | **18/20** | 0.717 |
| gpt-5.6-sol | 128 | b32 | 26 | 2/20 | **2/20** | 0.018 |
| gpt-5.6-terra | 64 | hex | 16 | 12/20 | **13/20** | 0.442 |
| gpt-5.6-terra | 64 | b32 | 13 | 3/20 | **3/20** | 0.042 |
| gpt-5.6-terra | 128 | hex | 32 | 9/20 | **9/20** | 0.259 |
| gpt-5.6-terra | 128 | b32 | 26 | 0/20 | **0/20** | 0.000 |


## 22. Cross-provider conclusions

Generated by `compare.py` - every table is paired on identical images.

```
Claude vs GPT-5.6 - identical images; image DELIVERY differs (path+Read vs attached -i)

=== BITMAP LADDER (identical images) ===
model             mean lookup  mean legible  s/image   n
opus                    61.7%          0.79       43  15
sonnet                  61.7%          0.68      104  15
gpt-5.6-terra           31.7%          0.72       18  15
gpt-5.6-luna            36.7%          0.62       19  15
gpt-5.6-sol             55.0%          0.80       20  15

=== LEGIBILITY ONLY, paired vs opus (first/last lines, no row counting) ===
  gpt-5.6-terra    n= 26  opus 0.606 -> terra 0.426   delta -0.180   Wilcoxon p=0.0266
  gpt-5.6-luna     n= 26  opus 0.606 -> luna 0.480   delta -0.126   Wilcoxon p=0.0296
  gpt-5.6-sol      n= 26  opus 0.606 -> sol 0.761   delta +0.155   Wilcoxon p=0.0094

=== WHOLE 56-CHAR RECORDS, high-entropy payload, base matrix ===
model               exact   rate   95% LB  bind disp=0
opus                 0/18   0.00    0.000         6/24
gpt-5.6-terra        4/18   0.22    0.080        12/24
gpt-5.6-luna         2/18   0.11    0.020         9/24
gpt-5.6-sol          3/18   0.17    0.047        17/24

=== SHORT SPANS n=20 (literal = reading + length compliance; prefix = reading) ===
model           alpha    metric        sp8    sp16    sp32    sp51    total   95% LB
gpt-5.6-luna    crock32  literal      7/20    4/20    0/20    0/20    11/80    0.079
gpt-5.6-luna    crock32  prefix       7/20    7/20    1/20    0/20    15/80    0.119
gpt-5.6-luna    hex      literal     14/20   14/20    4/20   12/20    44/80    0.452
gpt-5.6-luna    hex      prefix      17/20   16/20   10/20   12/20    55/80    0.592
gpt-5.6-sol     crock32  literal      2/20    8/20    6/20    1/20    17/80    0.140
gpt-5.6-sol     crock32  prefix       2/20    8/20    6/20    1/20    17/80    0.140
gpt-5.6-sol     hex      literal      8/20   14/20   17/20   16/20    55/80    0.592
gpt-5.6-sol     hex      prefix      18/20   19/20   20/20   16/20    73/80    0.842
gpt-5.6-terra   crock32  literal     10/20    5/20    2/20    0/20    17/80    0.140
gpt-5.6-terra   crock32  prefix      11/20    5/20    2/20    0/20    18/80    0.151
gpt-5.6-terra   hex      literal     17/20   14/20   12/20   14/20    57/80    0.618
gpt-5.6-terra   hex      prefix      18/20   14/20   17/20   14/20    63/80    0.698
opus            b64      literal       0/3     0/3     0/3     0/3     0/12    0.000
opus            b64      prefix        0/3     0/3     0/3     0/3     0/12    0.000
opus            crock32  literal      9/20    7/20    4/20    0/20    20/80    0.172
opus            crock32  prefix       9/20    7/20    4/20    0/20    20/80    0.172
opus            hex      literal     18/20   16/20   15/20    0/20    49/80    0.515
opus            hex      prefix      18/20   16/20   15/20    0/20    49/80    0.515

=== DELIMITED EQUAL-INFORMATION FIELDS n=20 (decoded value equality) ===
model             bits   enc  chars   literal     value   95% LB
gpt-5.6-luna        64   hex     16     10/20     10/20    0.302
gpt-5.6-luna        64   b32     13      4/20      4/20    0.071
gpt-5.6-luna       128   hex     32      7/20      7/20    0.177
gpt-5.6-luna       128   b32     26      1/20      1/20    0.003
gpt-5.6-sol         64   hex     16     17/20     17/20    0.656
gpt-5.6-sol         64   b32     13     15/20     16/20    0.599
gpt-5.6-sol        128   hex     32     18/20     18/20    0.717
gpt-5.6-sol        128   b32     26      2/20      2/20    0.018
gpt-5.6-terra       64   hex     16     12/20     13/20    0.442
gpt-5.6-terra       64   b32     13      3/20      3/20    0.042
gpt-5.6-terra      128   hex     32      9/20      9/20    0.259
gpt-5.6-terra      128   b32     26      0/20      0/20    0.000
opus                64   hex     16     20/20     20/20    0.861
opus                64   b32     13     14/20     19/20    0.784
opus               128   hex     32     16/20     16/20    0.599
opus               128   b32     26     12/20     15/20    0.544
```

### What holds up

* **Sol reads dense glyphs better than Opus - but its siblings do not.** 26 paired
  legibility conditions on byte-identical files, addressing removed:

  | model | opus -> model | delta | Wilcoxon p |
  |---|---|---|---|
  | **gpt-5.6-sol** | 0.606 -> **0.761** | **+0.155** | **0.0094** |
  | gpt-5.6-luna | 0.606 -> 0.480 | -0.126 | 0.0296 |
  | gpt-5.6-terra | 0.606 -> 0.426 | -0.180 | 0.0266 |

  Only sol beats Claude; **luna and terra are both significantly worse**. Any claim of
  the form 'GPT-5.6 reads dense text better' is false as a family statement - the
  geometry is family-wide, the capability is not.
* **The long-span failure regime is Claude-specific.** Claude scored 0/20 at a
  51-character span - a cliff attributed in section 4 to counting or output stamina.
  Both 5.6 models tested read it: sol 16/20, luna 12/20. So that cliff is a property of
  Claude, not of vision models, and section 4's 'separate failure regime' should be read
  as provider-specific. Sol also hits 20/20 prefix-correct at 32 characters - a 128-bit
  handle read byte-exact, twenty times out of twenty.
* **Sol binds rows far better.** 17/24 against Claude's 6/24 on the same probe. Row
  addressing was *the* blocker on the Claude side and the entire reason E-BIND-1 routes
  around it.
* **The advantage is specific to unguessable payload.** On prose Claude is equal or
  better. That is consistent with the redundancy-credit finding in section 3: Claude
  leans on language priors to repair glyphs, and those priors do nothing for a hash.

### What does not

* **The whole-record comparison is underpowered.** 3/18 against 0/18 sounds decisive and
  is not: Fisher p = 0.229, CIs overlap, and pooling 3 cells x 3 payloads is exactly the
  heterogeneous pooling criticised in section 5. The span and legibility probes at n=20
  carry the argument; the confirm matrix does not.
* **Sol follows length instructions worse.** Its literal span score is 55/80 against
  Claude's 49/80, but only because prefix-tolerance rescues 18 answers where it returned
  N+1 characters. Claude cuts at exactly N every time. If your parser is strict, that
  difference is real and costs you.
* **Two of the three 5.6 models are worse than Claude at this task.** Ladder lookup:
  sol 55%, luna 37%, terra 32% against opus 39% and sonnet 62%. Picking 'the 5.6 line'
  without picking sol specifically would be a downgrade.

### Choosing between them

| if you care about | pick | why |
|---|---|---|
| cost per character | **gpt-5.6** | 853 vs 784 px/token, +8.8% at every cell |
| characters per request | **claude** | 1932x1932 page holds 74,498 vs 51,200 |
| reading very dense glyphs | **gpt-5.6-sol** | legibility 0.761 vs 0.606; reads 21.3 ch/token where nothing else does |
| **recovering an exact handle** | **claude** | delimited fields 70/80 vs sol 53/80, luna 22/80 |
| **resolving the right record** | **claude** | E-BIND-1 20/20 vs sol 17/20, luna 1/20 |
| **knowing when there is no answer** | **claude** | sol false-accepts ~38% of no-answer queries; claude 0% |
| exact adherence to output format | **claude** | cuts at exactly N; sol over-returns |
| latency | **gpt-5.6** | 20 s/image against 43 (opus) and 104 (sonnet) |

**The two rankings genuinely disagree.** Sol reads glyphs better than Claude; Claude
executes the retrieval protocol better than sol. If the job is 'squeeze the most
readable text into the fewest tokens', sol wins. If the job is 'return the correct
record, or admit there isn't one', Claude wins - and that is the job an agent actually
has.

Neither is safe for unverified handles. Every conclusion in Part I about checksums,
canonical fetch and provenance binding applies unchanged to both providers - and sol's
no-answer false-accept rate makes post-fetch reconciliation *more* necessary, not less.


## 23. E-BIND-1: resolving the correct canonical record

Semantic query, exact text page/block labels outside the bitmap, a 64-bit keyed
codeword inside it, canonical fetch and gold scoring. Outcome partitions are separate
for answer-present (`C`/`D`/`W`/`A`/`P`) and no-answer (`N`/`W0`/`D0`/`P0`) queries,
because abstention is correct in one and failure in the other.

| model | stage | carrier | C | D | **W** | A | N | **W0** | D0 | P |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | preflight | image/hex | 1 | 14 | **0** | 5 | 5 | **0** | 0 | 0 |
| gpt-5.6-luna | preflight | image/b32 | 0 | 14 | **0** | 6 | 4 | **0** | 1 | 0 |
| gpt-5.6-luna | preflight | text/hex | 20 | 0 | **0** | 0 | 5 | **0** | 0 | 0 |
| gpt-5.6-sol | preflight | image/hex | 17 | 3 | **0** | 0 | 0 | **3** | 2 | 0 |
| gpt-5.6-sol | preflight | image/b32 | 4 | 15 | **0** | 1 | 0 | **0** | 5 | 0 |
| gpt-5.6-sol | preflight | text/hex | 20 | 0 | **0** | 0 | 5 | **0** | 0 | 0 |
| gpt-5.6-sol | stage1 | image/hex | 190 | 102 | **8** | 0 | 2 | **22** | 36 | 0 |
| gpt-5.6-sol | stage1b32 | image/b32 | 29 | 69 | **2** | 0 | 6 | **1** | 13 | 0 |
| gpt-5.6-sol | stage1text | text/hex | 99 | 1 | **0** | 0 | 20 | **0** | 0 | 0 |
| gpt-5.6-terra | preflight | image/hex | 8 | 11 | **0** | 1 | 5 | **0** | 0 | 0 |
| gpt-5.6-terra | preflight | image/b32 | 2 | 18 | **0** | 0 | 3 | **0** | 2 | 0 |
| gpt-5.6-terra | preflight | text/hex | 20 | 0 | **0** | 0 | 5 | **0** | 0 | 0 |
| opus | preflight | image/hex | 20 | 0 | **0** | 0 | 5 | **0** | 0 | 0 |
| opus | preflight | image/b32 | 17 | 3 | **0** | 0 | 5 | **0** | 0 | 0 |
| opus | preflight | text/hex | 20 | 0 | **0** | 0 | 5 | **0** | 0 | 0 |

### The text ceiling isolates optics from semantics

Every model scores **20/20 C and 5/5 N on the text carrier** - identical corpus,
identical queries, identical distractors, records supplied as text instead of pixels.
So semantic selection, abstention and protocol compliance are all intact in all of
them, and the entire spread on the image carrier (opus 20, sol 17, luna 1) is optical.
Without this control a score of 1/20 could not be separated into 'cannot read' versus
'cannot find', and luna would look semantically broken when it is not.

The preflight also separates *reading* from *calibration* cleanly: terra (8/20 C) and
luna (1/20 C) read much worse than sol (17/20 C) yet both abstain correctly 5/5, while
sol abstains 0/5. Confidence and competence are independent here.

Note the ranking inverts against legibility: sol reads glyphs better than opus
(section 19) yet opus wins the protocol task, because E-BIND-1 additionally requires
exact code transcription, correct block selection and correct abstention - the three
things sol is weaker at.

### False-accept rates with bounds

| model | stage | carrier | present n | **W** | W rate | 95% UB | no-answer n | **W0** | W0 rate |
|---|---|---|---|---|---|---|---|---|---|
| gpt-5.6-luna | preflight | image | 20 | **0** | 0.000 | 0.139 | 5 | **0** | 0.000 |
| gpt-5.6-luna | preflight | text | 20 | **0** | 0.000 | 0.139 | 5 | **0** | 0.000 |
| gpt-5.6-sol | preflight | image | 20 | **0** | 0.000 | 0.139 | 5 | **3** | 0.600 |
| gpt-5.6-sol | preflight | text | 20 | **0** | 0.000 | 0.139 | 5 | **0** | 0.000 |
| gpt-5.6-sol | stage1 | image | 300 | **8** | 0.027 | 0.048 | 60 | **22** | 0.367 |
| gpt-5.6-sol | stage1text | text | 100 | **0** | 0.000 | 0.030 | 20 | **0** | 0.000 |
| gpt-5.6-terra | preflight | image | 20 | **0** | 0.000 | 0.139 | 5 | **0** | 0.000 |
| gpt-5.6-terra | preflight | text | 20 | **0** | 0.000 | 0.139 | 5 | **0** | 0.000 |
| opus | preflight | image | 20 | **0** | 0.000 | 0.139 | 5 | **0** | 0.000 |
| opus | preflight | text | 20 | **0** | 0.000 | 0.139 | 5 | **0** | 0.000 |

### Stage 1 at scale, and why the bounds above are optimistic

360 calls on 12 blocks / 144 records. Answer-present: **C 63.3%**, D 34.0%,
**W 2.7%**, A 0%. No-answer: **N 3.3%**, **W0 36.7%**, D0 60%.

But those 360 calls are **120 unique items x 3 passes**, not 360
independent trials. 78/120 items (65%) returned the identical outcome on all three passes, and only
**5 distinct items** ever produced a `W`. The repeated passes estimate
call-level variance conditional on the item set; they do not multiply the item
sample. Per-item n is 120, so every bound in the table above is
optimistic - the correct denominator for generalising to new items is the item
count, not the call count.

### The text ceiling isolates the deficit as purely optical

The same 120 queries, same records, same distractors, run with the archive supplied
as text instead of images:

| carrier | resolution | correct abstention |
|---|---|---|
| text | **99.0%** (99/100) | **100%** (20/20) |
| image, hex | 63.3% (190/300) | 3.3% (2/60) |

Fisher p = 1.3e-15 for resolution and 6.5e-17 for abstention. Semantic selection,
protocol compliance and calibration are all **intact** - sol finds the right record
and correctly reports absence when it can read the archive. Every point of the
36-point resolution gap and the whole abstention collapse are attributable to
reading pixels rather than to any reasoning failure.

**Sol never abstains on the image carrier.** `A` = 0 across all 300 answer-present
calls and `N` = 2/60 on no-answer. It always produces something; the keyed tag then
rejects 34% of it. On more than a third of questions with no answer at all, sol
returns a validating codeword for a real - but wrong - record. That failure is
invisible to any checksum.


A `W` is a *valid* codeword for a *real* record, returned for a different record's
question. No checksum or tag width detects it - only reconciliation after canonical
fetch. This is the number that decides whether an optical lookup tier is safe.

### Sol's abstention breaks only on the image carrier

Sol returned `N` (correct NO_MATCH) **5/5** on the text carrier and **0/5** on the
image carrier, where all five no-answer queries produced something instead: 3 `W0` plus
2 `D0`. Claude abstained correctly in both. So this is not a general calibration
failure - sol knows when text contains no answer, and stops knowing when the same
records arrive as pixels. That is the dangerous direction, because `W0` is a *valid*
codeword for a *real* record returned for a question with no answer.

Sol's base32 collapse also reappears here (4/20 against Claude's 17/20) - the third
independent confirmation after the span and delimited-field probes.

`W` is the load-bearing number: a wrong canonical record accepted as correct. `W0`
is its no-answer twin - a real record returned for a query that has no answer. A
checksum cannot catch either, because the returned codeword is a valid codeword for a
real record; only reconciliation after fetch can.


## 24. Sol-native frontier (32-grid canvases)

Sections 18-21 deliberately reuse Claude's 28-grid images so the stimuli are
identical. Those canvases are not optimal for a 32px patch grid, so this section
regenerates the packing series on zero-waste 32-grid canvases and re-measures. The
`ch/token` column here is **sol's own** rate (1024/1.2 per patch), not Claude's.

| cell | canvas | ch/token (sol) | legibility |
|---|---|---|---|
| 5x6 | 1120x672 | 27.8 | 0.00 |
| 4x9 | 864x864 | 23.5 | 0.00 |
| 6x6 | 1344x672 | 23.1 | 0.00 |
| 5x8 | 1120x896 | 20.9 | 0.20 |
| 4x10 | 896x1120 | 20.8 | 0.00 |
| 5x9 | 800x864 | 18.8 | 0.00 |
| 6x8 | 672x896 | 17.3 | 0.27 |
| 4x12 | 672x1344 | 17.3 | 0.96 |
| 5x10 | 800x1120 | 16.5 | 0.98 |
| 6x9 | 864x864 | 15.8 | 0.00 |
| 6x9 | 384x1152 | 15.8 | 0.86 |
| 7x8 | 896x896 | 14.7 | 0.64 |
| 6x10 | 384x1120 | 14.2 | 0.95 |
| 6x10 | 672x1120 | 13.8 | 0.93 |
| 8x8 | 896x896 | 12.7 | 0.73 |
| 6x13 | 384x1248 | 10.9 | 0.94 |
| 6x13 | 768x1248 | 10.6 | 0.94 |
| 7x14 | 448x1568 | 8.7 | 0.97 |
| 8x16 | 448x1568 | 6.4 | 0.99 |
| 8x16 | 448x1600 | 6.4 | 0.99 |
| 9x18 | 576x1440 | 5.3 | 1.00 |
| 10x20 | 640x1440 | 4.3 | 0.98 |
| 12x24 | 672x1344 | 2.9 | 0.99 |

Densest 32-grid cell reaching legibility >= 0.95: **5x9** at **18.4 chars per sol token** (800x864, legibility 0.98).


---

## 25. Method: the GPT-5.6 arm

Every 5.6 call goes through `providers.py`, which normalises two genuinely different
CLIs behind one interface:

```
codex exec --ephemeral --ignore-user-config --skip-git-repo-check \
           -s read-only -m <exact-slug> -c model_reasoning_effort=low \
           -i <image> --json          # prompt on STDIN, never as a positional arg
```

| flag | why it is mandatory |
|---|---|
| `--ignore-user-config` | the local `config.toml` sets `model_reasoning_effort = high` and a personality; without this every run is silently a different experiment |
| `-c model_reasoning_effort=low` | effort is a free variable the Claude arm never controlled; pinned and recorded |
| `--ephemeral` | no session carry-over between trials |
| `-s read-only` | no side effects |
| `-i <image>` | the image is **attached**, so the model never receives a path |
| prompt on stdin | `-i` is variadic and eats a trailing positional prompt |

**The one asymmetry that cannot be removed.** `codex exec` attaches images; `claude -p`
receives a path and fetches with the Read tool. Stimuli are byte-identical, delivery is
not. A side effect is that the crop-and-zoom confound which forced `--disallowedTools` on
the Claude side is *structurally impossible* on codex - there is no path to act on - so
the codex arm is cleaner in that one respect and different in another. Every comparison
table says so.

Each result records the exact model slug (never `sol`/`opus`), effort, CLI version, prompt
SHA and image-delivery mode. Usage comes from the `turn.completed` event of the `--json`
stream - note that session files use a different shape (`token_count`), which cost one
debugging cycle.

### Call budget actually spent

| phase | calls |
|---|---|
| 1 geometry (3 models, rate + ceiling + bisect + adjacent) | ~90 |
| 2 ladder + legibility + confirm + extension (3 models) | ~460 |
| 3 spans hex + crock32 + delimited (3 models) + sol-native p32 | ~750 |
| 4 E-BIND-1 preflight (3 models) + Stage 1 (hex x3, b32, text) | ~800 |
| **total** | **~2,100** |

The plan budgeted ~2,230 on Claude timings; 5.6 calls run 5-20s against Claude's 10-280s,
so wall-clock was a fraction of the estimate.

---

## 26. Where this leaves things

**Solved.** Image-token geometry for both providers, zero-waste canvas construction, the
downscale ceilings (Claude 1932x1932 at 4,761 patches; GPT-5.6 1600x1600 at 2,500 patches
billing 1.2 tokens each), and the density frontier for every glyph cell on both grids.

**Measured, with the caveats stated.** Legibility, span reading, delimited-field recovery
and full protocol resolution for five models on byte-identical stimuli. Sol reads dense
glyphs best; Claude executes the retrieval protocol best; the two rankings disagree and
the disagreement is the useful finding.

**Established and unwelcome.** No configuration on either provider is safe for unverified
handles. Sol additionally returns a valid codeword for a real record on ~37% of questions
that have no answer, which no checksum can detect. Post-fetch semantic reconciliation is
not optional on either provider, and is more necessary on sol than on Claude.

**Still open.** `P(false accept)` for a *structured* codeword (32-bit index + 32-bit keyed
tag) rather than a uniform random value; the 2x2 grouping-by-delimiting design; equivalence
testing against a declared margin rather than null-hypothesis tests; effort sweeps on the
5.6 line, all of which ran pinned at `low`; and Arm B native-density page goodput, which
would need complete protocol pages rendered on each provider's own optimal canvas.

The single most useful next measurement is unchanged from section 13: a structured-codeword
false-accept rate with wrong-valid-handle decoys, on whichever provider is actually going
to be deployed - because that number, not chars-per-token, decides whether an optical
memory tier is viable.

