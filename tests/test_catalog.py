import csv
import json
import os
from datetime import datetime, timezone

import provenance as V
from validate_project import EXPECTED_IMAGES, ROOT, validate_catalog, validate_results


def test_catalog_files_hashes_and_manifest_agree():
    assert validate_catalog() == []
    with open(os.path.join(ROOT, 'ANSWER_KEY.json')) as f:
        catalog = json.load(f)
    assert catalog['schema_version'] == 'vision-text-catalog-v2'
    assert len(catalog['images']) == EXPECTED_IMAGES
    flagged = {name for name, row in catalog['images'].items()
               if row['exceeds_claude_measured_envelope']}
    assert flagged == {'E_degradations/oversized/E2_2240x2240_5MP.png'}


def test_result_envelope_schema_accepts_minimal_record():
    import jsonschema
    with open(os.path.join(ROOT, 'schemas', 'result-envelope.schema.json')) as f:
        schema = json.load(f)
    envelope = {
        'schema_version': V.RESULT_SCHEMA_VERSION,
        'manifest': {
            'schema_version': V.RESULT_SCHEMA_VERSION,
            'experiment': 'unit', 'model': 'exact-model-id', 'provider': 'test',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'effort': None, 'cli_version': None, 'harness_sha256': '0' * 16,
            'prompts': {}, 'images': [], 'environment': {},
        },
        'results': [],
    }
    jsonschema.validate(envelope, schema)


def test_paid_result_artifacts_refuse_implicit_overwrite(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text('{}')
    import pytest
    with pytest.raises(FileExistsError):
        V.require_new_output(str(path))
    V.require_new_output(str(path), overwrite=True)


def test_v2_provenance_semantics_reject_manifest_row_mismatch(tmp_path):
    (tmp_path / 'schemas').mkdir()
    with open(os.path.join(ROOT, 'schemas', 'result-envelope.schema.json')) as f:
        schema_text = f.read()
    (tmp_path / 'schemas' / 'result-envelope.schema.json').write_text(schema_text)
    envelope = {
        'schema_version': V.RESULT_SCHEMA_VERSION,
        'manifest': {
            'schema_version': V.RESULT_SCHEMA_VERSION, 'experiment': 'unit',
            'created_at': datetime.now(timezone.utc).isoformat(), 'model': 'exact-model',
            'provider': 'codex', 'effort': 'low', 'cli_version': 'unit',
            'harness_sha256': '0' * 16, 'prompts': {}, 'images': [], 'environment': {},
        },
        'results': [{
            'model': 'different-model', 'provider': 'codex', 'effort': 'low',
            'response': {'text': None, 'usage': {}, 'returncode': 1, 'stderr': '',
                         'stdout': '', 'argv': [], 'provider': 'codex',
                         'model': 'different-model', 'effort': 'low', 'n_images': 0},
        }],
    }
    V.dump_json(str(tmp_path / 'results_unit.json'), envelope)
    errors = validate_results(str(tmp_path))
    assert any('row 0 model disagrees with manifest' in e for e in errors)
    envelope['results'][0]['model'] = 'exact-model'
    envelope['results'][0]['response']['model'] = 'exact-model'
    V.dump_json(str(tmp_path / 'results_unit.json'), envelope)
    assert validate_results(str(tmp_path)) == []
