#!/usr/bin/env python3
"""Paired 2x2 grouping-by-delimiting experiment for exact hexadecimal fields."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import math
import os
import random
import re
import time

from gen_lib import bitmap, canvas, draw_lines
from delim_probe import decode_value, enc_hex
import providers as P
import provenance as V

ROOT = os.path.dirname(os.path.abspath(__file__))

PROMPT = ("{head}The FIRST line at the top contains a hexadecimal field after the label FIELD. "
          "{boundary} Transcribe the field value only. Spaces inserted inside the field are "
          "visual grouping and are not part of the value. Output STRICT JSON only: "
          '{{"field": "..."}}. If unresolvable use {{"field": "UNREADABLE"}}. Do not guess.')


def format_value(value, bits, grouped, delimited):
    raw = enc_hex(value, bits)
    shown = ' '.join(raw[i:i + 4] for i in range(0, len(raw), 4)) if grouped else raw
    return f'[{shown}]' if delimited else shown


def render(cell, bits, rep, grouped, delimited):
    m = re.match(r'(?:\D*)(\d+)x(\d+)', cell)
    cw, lh = int(m.group(1)), int(m.group(2))
    rng = random.Random(f'layout-v1|{cell}|{bits}|{rep}')  # condition excluded: fully paired
    values = [rng.getrandbits(bits) for _ in range(96)]
    f = bitmap(cell)
    lines = [f'FIELD {format_value(v, bits, grouped, delimited)}' for v in values]
    cols = max(len(x) for x in lines)
    uw = 28 * cw // math.gcd(28, cw)
    uh = 28 * lh // math.gcd(28, lh)
    w = uw * math.ceil(cols * cw / uw)
    h = uh * max(1, min(math.ceil(len(lines) * lh / uh), 1456 // uh))
    rows = h // lh
    lines, values = lines[:rows], values[:rows]
    im, d = canvas(w, h)
    draw_lines(d, f, lines, 0, 0, lh=lh)
    return im, lines, values, w, h


def run_one(job, model, effort):
    cell, bits, rep, grouped, delimited = job
    im, lines, values, w, h = render(cell, bits, rep, grouped, delimited)
    folder = os.path.join(ROOT, 'images', 'LAYOUT2X2')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'{cell}_{bits}_r{rep}_g{int(grouped)}d{int(delimited)}.png')
    im.save(path)
    boundary = ('The field is enclosed in square brackets; do not include them.' if delimited
                else f'The field has exactly {bits // 4} hexadecimal digits.')
    provider = P.provider_for(model)
    prompt = PROMPT.format(head=P.image_head(provider, path), boundary=boundary)
    t0 = time.time()
    response = P.run(prompt, model=model, images=P.images_for(provider, [path]),
                     effort=effort, timeout=900, cwd=ROOT)
    answer = P.parse_json_answer(response['text'])
    got = str((answer or {}).get('field') or '').strip() if isinstance(answer, dict) else ''
    value = None if got.upper() == 'UNREADABLE' else decode_value(got, 'hex', bits)
    want = enc_hex(values[0], bits)
    return {
        'item_id': f'{bits}-{rep}', 'cell': cell, 'bits': bits, 'rep': rep,
        'grouped': grouped, 'delimited': delimited, 'model': model,
        'provider': provider, 'effort': P.effective_effort(model, effort),
        'got': got, 'want': want,
        'literal': got == want, 'value': value == values[0],
        'abstain': got.upper() == 'UNREADABLE',
        'cer': 1 - difflib.SequenceMatcher(None, got, want).ratio(),
        'w': w, 'h': h, 'image_sha256': V.sha256_file(path),
        'seconds': round(time.time() - t0, 1), 'response': V.response_record(response),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cell', default='6x13')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--effort', default='low')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--allow-mutable-model-alias', action='store_true')
    a = ap.parse_args()
    if P.model_is_mutable_alias(a.model) and not a.allow_mutable_model_alias:
        ap.error('--model must be an exact immutable model id (or opt into an exploratory alias)')
    suffix = '.dry-run' if a.dry_run else ''
    path = os.path.join(ROOT, f'results_layout2x2_{a.cell}_{a.model}{suffix}.json')
    V.require_new_output(path, a.overwrite)
    jobs = [(a.cell, bits, rep, grouped, delimited)
            for bits in (64, 128) for rep in range(1, a.reps + 1)
            for grouped in (False, True) for delimited in (False, True)]
    if a.dry_run:
        # Rendering one representative of every condition validates geometry without calls.
        for job in [j for j in jobs if j[2] == 1]: render(*job)
        rows = []
    else:
        with ThreadPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(lambda job: run_one(job, a.model, a.effort), jobs))
    provider = P.provider_for(a.model)
    manifest = V.manifest(
        experiment='layout-grouping-delimiting-2x2-v1', model=a.model, provider=provider,
        effort=a.effort, cli_version=P.cli_version(provider), harness_path=__file__,
        prompts={'field': PROMPT}, cell=a.cell, reps=a.reps, bits=[64, 128],
        paired_seed_excludes=['grouped', 'delimited'], dry_run=a.dry_run)
    V.dump_json(path, {'schema_version': V.RESULT_SCHEMA_VERSION,
                       'manifest': manifest, 'results': rows})
    print(f'{len(jobs)} planned calls; {len(rows)} completed -> {path}')


if __name__ == '__main__':
    main()
