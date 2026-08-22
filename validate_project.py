#!/usr/bin/env python3
"""Offline project invariants; safe for CI and preflight."""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import provenance as V
import providers as P

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPECTED_IMAGES = 161


def validate_catalog(root=ROOT):
    key_path = os.path.join(root, 'ANSWER_KEY.json')
    with open(key_path) as f: key = json.load(f)
    errors = []
    if key.get('schema_version') != 'vision-text-catalog-v2':
        errors.append('ANSWER_KEY.json is not catalog schema v2; run generate.py && pack.py')
    images = key.get('images') or {}
    if len(images) != EXPECTED_IMAGES:
        errors.append(f'catalog has {len(images)} entries, expected {EXPECTED_IMAGES}')
    for rel, meta in images.items():
        image = os.path.join(root, 'images', rel)
        truth = os.path.join(root, meta.get('groundtruth', ''))
        for label, path, expected in (
            ('image', image, meta.get('image_sha256')),
            ('groundtruth', truth, meta.get('groundtruth_sha256')),
        ):
            if not os.path.isfile(path): errors.append(f'missing {label}: {path}'); continue
            if not expected: errors.append(f'missing {label} hash: {rel}'); continue
            got = V.sha256_file(path)
            if got != expected: errors.append(f'{label} hash mismatch: {rel}')
    manifest = os.path.join(root, 'manifest.csv')
    with open(manifest, newline='') as f: rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_IMAGES:
        errors.append(f'manifest has {len(rows)} rows, expected {EXPECTED_IMAGES}')
    if {r['file'] for r in rows} != set(images):
        errors.append('manifest and answer-key file sets differ')
    return errors


def validate_results(root=ROOT):
    errors = []
    schema_path = os.path.join(root, 'schemas', 'result-envelope.schema.json')
    try:
        import jsonschema
        with open(schema_path) as f: schema = json.load(f)
    except (ImportError, OSError) as e:
        return [f'cannot load result schema validator: {e}']
    for path in sorted(glob.glob(os.path.join(root, 'results_*.json'))):
        try:
            with open(path) as f: blob = json.load(f)
        except Exception as e:
            errors.append(f'invalid JSON {os.path.basename(path)}: {e}'); continue
        if isinstance(blob, dict) and blob.get('schema_version') == V.RESULT_SCHEMA_VERSION:
            validator = jsonschema.Draft202012Validator(
                schema, format_checker=jsonschema.FormatChecker())
            for e in validator.iter_errors(blob):
                errors.append(f'result schema error {os.path.basename(path)}: {e.message}')
            manifest = blob.get('manifest') or {}
            model = manifest.get('model')
            if P.model_is_mutable_alias(model):
                errors.append(f'v2 result uses mutable model alias: {os.path.basename(path)}')
            for i, row in enumerate(blob.get('results') or []):
                response = row.get('response') or {}
                for field in ('model', 'provider', 'effort'):
                    expected = manifest.get(field)
                    if row.get(field) != expected:
                        errors.append(f'{os.path.basename(path)} row {i} {field} '
                                      f'disagrees with manifest')
                    if response.get(field) != expected:
                        errors.append(f'{os.path.basename(path)} row {i} response {field} '
                                      f'disagrees with manifest')
        if os.path.basename(path).startswith('results_confirm-v1-ext_') and isinstance(blob, dict):
            rows = V.result_rows(blob)
            actual_cells = sorted({r.get('font') for r in rows})
            actual_payloads = sorted({r.get('payload') for r in rows})
            if sorted(blob.get('manifest', {}).get('cells', [])) != actual_cells:
                errors.append(f'extension manifest cells disagree: {os.path.basename(path)}')
            if sorted(blob.get('manifest', {}).get('payloads', [])) != actual_payloads:
                errors.append(f'extension manifest payloads disagree: {os.path.basename(path)}')
    return errors


def validate_rebuild():
    """Rebuild in a temporary copy and require the authoritative artifact hashes to match."""
    with tempfile.TemporaryDirectory(prefix='vision-text-rebuild-') as tmp:
        payload = subprocess.check_output(['git', 'archive', 'HEAD'], cwd=ROOT)
        subprocess.run(['tar', '-xf', '-', '-C', tmp], input=payload, check=True)
        # Overlay current tooling, including new untracked files, when run before commit.
        for src in glob.glob(os.path.join(ROOT, '*.py')):
            shutil.copy2(src, os.path.join(tmp, os.path.basename(src)))
        if os.path.isdir(os.path.join(ROOT, 'schemas')):
            shutil.copytree(os.path.join(ROOT, 'schemas'), os.path.join(tmp, 'schemas'),
                            dirs_exist_ok=True)
        subprocess.run([sys.executable, 'generate.py'], cwd=tmp, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, 'pack.py'], cwd=tmp, check=True,
                       stdout=subprocess.DEVNULL)
        errors = validate_catalog(tmp)
        with open(os.path.join(ROOT, 'ANSWER_KEY.json')) as f: expected = json.load(f)['images']
        with open(os.path.join(tmp, 'ANSWER_KEY.json')) as f: rebuilt = json.load(f)['images']
        for rel in sorted(set(expected) | set(rebuilt)):
            if rel not in expected or rel not in rebuilt:
                errors.append(f'rebuilt catalog file-set mismatch: {rel}'); continue
            for field in ('image_sha256', 'groundtruth_sha256'):
                if expected[rel].get(field) != rebuilt[rel].get(field):
                    errors.append(f'rebuild hash mismatch for {rel} ({field})')
        return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true', help='also perform the slower isolated rebuild')
    a = ap.parse_args()
    errors = validate_catalog() + validate_results()
    if a.rebuild: errors += [f'rebuild: {x}' for x in validate_rebuild()]
    if errors:
        print('\n'.join(f'ERROR: {e}' for e in errors), file=sys.stderr)
        raise SystemExit(1)
    print(f'project invariants: OK ({EXPECTED_IMAGES} catalog entries)')


if __name__ == '__main__': main()
