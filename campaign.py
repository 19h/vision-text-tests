#!/usr/bin/env python3
"""Plan or explicitly execute the remaining confirmatory benchmark campaigns."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

import providers as P
import provenance as V

ROOT = os.path.dirname(os.path.abspath(__file__))


def command(script, *args):
    return [sys.executable, os.path.join(ROOT, script), *map(str, args)]


def artifact_complete(stage, name, root=ROOT):
    path = os.path.join(root, name)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    if not name.endswith('.json'):
        return True
    try:
        with open(path) as f: value = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if name.startswith('results_'):
        return bool(
            isinstance(value, dict) and value.get('schema_version') == V.RESULT_SCHEMA_VERSION
            and isinstance(value.get('results'), list)
            and not (value.get('manifest') or {}).get('dry_run')
            and len(value['results']) == stage['estimated_calls'])
    return isinstance(value, dict)


def stage_complete(stage, root=ROOT):
    return bool(stage.get('outputs')) and all(
        artifact_complete(stage, name, root) for name in stage['outputs'])


def plan(model):
    stages = [
        {
            'name': 'heldout_stage1_structured_32_32', 'estimated_calls': 480,
            'purpose': 'paired image/text protocol estimate on 120 unique held-out items',
            'command': command('ebind1.py', '--model', model, '--stage', 'heldout-v1',
                '--arm', 'A', '--present', 100, '--absent', 20, '--blocks', 12,
                '--per-block', 12, '--reps', 3, '--text-reps', 1,
                '--carriers', 'image', 'text', '--encs', 'hex',
                '--index-bits', 32, '--tag-bits', 32),
        },
        {
            'name': 'arm_b_native_goodput', 'estimated_calls': 240,
            'purpose': 'native-grid hex/base32 architectural goodput on the same item namespace',
            'command': command('ebind1.py', '--model', model, '--stage', 'heldout-v1',
                '--arm', 'B', '--present', 100, '--absent', 20, '--blocks', 12,
                '--per-block', 12, '--reps', 1, '--text-reps', 1,
                '--carriers', 'image', '--encs', 'hex', 'b32',
                '--index-bits', 32, '--tag-bits', 32),
        },
        {
            'name': 'analyze_heldout_protocol', 'estimated_calls': 0,
            'purpose': 'item/page-clustered Arm-A/Arm-B and paired-carrier report',
            'command': command('analyze_ebind.py',
                f'results_ebind1_heldout-v1_A_{model}.json',
                f'results_ebind1_heldout-v1_B_{model}.json',
                '--json-out', f'analysis_ebind_heldout_{model}.json',
                '--markdown-out', f'analysis_ebind_heldout_{model}.md'),
        },
        {
            'name': 'semantic_verifier_wrong_valid', 'estimated_calls': 220,
            'purpose': 'fail-closed post-fetch reconciliation on correct and deliberately wrong-valid records',
            'command': command('verifier_probe.py', '--model', model, '--stage', 'heldout-v1',
                '--present', 100, '--absent', 20, '--blocks', 12, '--per-block', 12),
        },
        {
            'name': 'analyze_semantic_verifier', 'estimated_calls': 0,
            'purpose': 'clustered sensitivity, false-accept, and paired-discrimination report',
            'command': command('analyze_verifier.py',
                f'results_verifier_heldout-v1_{model}.json',
                '--json-out', f'analysis_verifier_heldout_{model}.json',
                '--markdown-out', f'analysis_verifier_heldout_{model}.md'),
        },
        {
            'name': 'grouping_delimiting_2x2', 'estimated_calls': 160,
            'purpose': 'paired factorial separation of grouping and visible boundaries',
            'command': command('layout_probe.py', '--model', model, '--cell', '6x13', '--reps', 20),
        },
        {
            'name': 'analyze_grouping_delimiting', 'estimated_calls': 0,
            'purpose': 'paired factorial effect intervals against the declared margin',
            'command': command('analyze_layout.py', f'results_layout2x2_6x13_{model}.json',
                               '--out', f'analysis_layout2x2_{model}.json'),
        },
        {
            'name': 'encoding_equivalence_d010', 'estimated_calls': 744,
            'purpose': 'paired hex/base32 equivalence at d=0.10; n=186 powers 30% discordance at 80%',
            'command': command('delim_probe.py', '--model', model, '--cell', '6x13',
                               '--reps', 186, '--tag', '_equivalence-d010'),
        },
        {
            'name': 'analyze_encoding_equivalence', 'estimated_calls': 0,
            'purpose': 'paired d=0.10 equivalence decision',
            'command': command('analyze_encoding.py',
                f'results_delim_6x13_equivalence-d010_{model}.json', '--margin', .10,
                '--out', f'analysis_encoding_equivalence_{model}.json'),
        },
        {
            'name': 'large_cell_delimited_9x18', 'estimated_calls': 80,
            'purpose': 'short exact fields at a larger cell',
            'command': command('delim_probe.py', '--model', model, '--cell', '9x18', '--reps', 20),
        },
        {
            'name': 'large_cell_delimited_12x24', 'estimated_calls': 80,
            'purpose': 'short exact fields at the largest tested cell',
            'command': command('delim_probe.py', '--model', model, '--cell', '12x24', '--reps', 20),
        },
    ]
    for effort in ('low', 'medium', 'high'):
        stages.append({
            'name': f'effort_{effort}_delimited', 'estimated_calls': 80,
            'purpose': 'effort sensitivity at the frozen 6x13 condition',
            'command': command('delim_probe.py', '--model', model, '--cell', '6x13',
                               '--reps', 20, '--effort', effort, '--tag', f'_effort-{effort}'),
        })
    stages.append({
        'name': 'rectangular_geometry', 'estimated_calls': 21,
        'purpose': 'map moderate aspect ratios and long-edge behavior',
        'command': command('geometry.py', '--model', model, '--tag', 'rectangles',
                           '--sizes', 896, '--reps', 3, '--patch', 32,
                           '--rectangles', '1344x896', '896x1344', '1600x800',
                           '800x1600', '1536x1024', '1024x1536'),
    })
    outputs = {
        'heldout_stage1_structured_32_32': [f'results_ebind1_heldout-v1_A_{model}.json'],
        'arm_b_native_goodput': [f'results_ebind1_heldout-v1_B_{model}.json'],
        'analyze_heldout_protocol': [f'analysis_ebind_heldout_{model}.json',
                                     f'analysis_ebind_heldout_{model}.md'],
        'semantic_verifier_wrong_valid': [f'results_verifier_heldout-v1_{model}.json'],
        'analyze_semantic_verifier': [f'analysis_verifier_heldout_{model}.json',
                                      f'analysis_verifier_heldout_{model}.md'],
        'grouping_delimiting_2x2': [f'results_layout2x2_6x13_{model}.json'],
        'analyze_grouping_delimiting': [f'analysis_layout2x2_{model}.json'],
        'encoding_equivalence_d010': [f'results_delim_6x13_equivalence-d010_{model}.json'],
        'analyze_encoding_equivalence': [f'analysis_encoding_equivalence_{model}.json'],
        'large_cell_delimited_9x18': [f'results_delim_9x18_{model}.json'],
        'large_cell_delimited_12x24': [f'results_delim_12x24_{model}.json'],
        'effort_low_delimited': [f'results_delim_6x13_effort-low_{model}.json'],
        'effort_medium_delimited': [f'results_delim_6x13_effort-medium_{model}.json'],
        'effort_high_delimited': [f'results_delim_6x13_effort-high_{model}.json'],
        'rectangular_geometry': [f'results_geometry_rectangles_{model}.json'],
    }
    for stage in stages:
        stage['outputs'] = outputs[stage['name']]
    return stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default='campaign-plan.json')
    ap.add_argument('--execute', action='store_true',
                    help='make paid external model calls; omitted by default intentionally')
    ap.add_argument('--resume', action='store_true',
                    help='with --execute, skip stages whose complete output set already exists')
    ap.add_argument('--only', nargs='*', default=[])
    a = ap.parse_args()
    if P.model_is_mutable_alias(a.model):
        ap.error('--model must be an exact immutable model id')
    if os.path.basename(a.model) != a.model or a.model in ('.', '..'):
        ap.error('--model may not contain path separators')
    stages = plan(a.model)
    if a.only:
        unknown = set(a.only) - {s['name'] for s in stages}
        if unknown: ap.error(f"unknown --only stages: {', '.join(sorted(unknown))}")
        stages = [s for s in stages if s['name'] in set(a.only)]
    if a.resume and not a.execute:
        ap.error('--resume requires --execute')
    if a.execute:
        partial = []
        conflicts = []
        for stage in stages:
            present = [name for name in stage['outputs']
                       if os.path.exists(os.path.join(ROOT, name))]
            if present and not stage_complete(stage):
                partial.append(f"{stage['name']}: {', '.join(present)}")
            elif present and not a.resume:
                conflicts.extend(present)
        if partial:
            ap.error('partial stage outputs require manual audit before resuming: ' + '; '.join(partial))
        if conflicts:
            ap.error('outputs already exist; use --resume to skip complete stages: ' +
                     ', '.join(conflicts))
    record = {'schema_version': 'vision-text-campaign-v1', 'model': a.model,
              'estimated_calls': sum(s['estimated_calls'] for s in stages), 'stages': stages}
    V.dump_json(os.path.join(ROOT, a.out), record)
    for stage in stages:
        print(f"{stage['name']}: ~{stage['estimated_calls']} calls")
        print('  ' + shlex.join(stage['command']))
        if a.execute:
            if a.resume and stage_complete(stage):
                print('  skipped: complete output set already exists')
                continue
            subprocess.run(stage['command'], cwd=ROOT, check=True)
    if not a.execute:
        print(f"planned {record['estimated_calls']} calls; no external calls made (pass --execute explicitly)")


if __name__ == '__main__':
    main()
