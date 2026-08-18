# Prompts

Use one prompt per axis; keep it identical across the whole ladder so results are comparable.

## 1. Transcription floor (series A, B, C)

> Transcribe the text in this image exactly, preserving line breaks and the leading
> 4-digit line numbers. Do not guess: if a line or character is not legible, write
> `[UNREADABLE]` in place of it rather than inventing plausible text. At the end,
> state how many lines you transcribed and how many were unreadable.

Score: per-line exact match, and character error rate against `groundtruth/<name>.txt`.
The corpus is random, so a confabulated line is trivially detectable.

## 2. Addressed lookup (cheap, high-signal)

> What is the 5-character code on line 0421? Answer with the code only.

Cheaper than full transcription and it separates *reading* from *scanning*: models often
read fine but lose track of line position on dense pages.

## 3. Needle retrieval

> One line contains a passphrase between `!!` markers. What is the passphrase, and on
> which line number does it appear?

> How many times does the word ZEPHYR appear in this image? List the line numbers.

## 4. Comprehension, not OCR (series C, F3)

> Summarise what this passage describes, then list every "the tally stood at N units
> against a quota of M" pair you can read, with the inspector code and time for each.

## 5. Arithmetic over what was read (F1 table, F4 receipt, F6 spreadsheet)

> Read the QTY column, add it yourself, and say whether the printed TOTAL is correct.

> What value is in cell F17? What is the sum of column C?

Tests whether the numbers were read individually or the total was simply copied.

## 6. Legibility floor in one request (series D)

> This image shows the same sentence rendered at increasing sizes. Transcribe every line
> from the top. For each line you cannot read, say so explicitly instead of guessing.

## 7. Within-image cascade (series G)

> This page has several text sizes. For each size tier, report its bracketed CODE.
> If a tier is illegible, say which one and why.

## Controls worth running

* Same image, PNG vs `jpeg30` - isolates compression damage from size.
* `A` vs `B` at equal glyph height - isolates anti-aliasing blur from glyph size.
* `E_*_00_baseline` vs `E_*_09_downup_50pct` - shows what a client-side resize costs you.
* `oversized/` at 2240x2240 - the provider downscales it; compare against the same content
  rendered at 896x896 to see how much is lost before the model ever sees it.
