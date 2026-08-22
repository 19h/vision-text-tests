import json

import pytest

import providers
import confirm
import geometry
from delim_probe import decode_value, enc_b32
from ebind1 import (build_corpus, classify, dec_hex as ebind_dec_hex, enc_hex, make_code, make_queries,
                    render_block, verify_code)
from gen_lib import geom, image_tokens


def test_nested_and_multiple_json_objects():
    assert providers.parse_json_answer('prefix {"a":{"b":1}} suffix') == {'a': {'b': 1}}
    assert providers.parse_json_answer('{"old":1}\nnoise\n{"new":2}') == {'new': 2}


def test_geometry_family_resolution():
    for model in ('gpt-5.6-sol', 'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-future'):
        assert geom(model)['patch'] == 32
        assert geom(model)['tokens_per_patch'] == 1.2
    assert geom('opus')['patch'] == 28
    assert providers.effective_effort('gpt-5.6-sol', 'high') == 'high'
    assert providers.effective_effort('claude-opus-4-1-20250805', 'high') is None


@pytest.mark.parametrize('model,expects_attachment', [
    ('gpt-5.6-sol', True),
    ('claude-opus-4-1-20250805', False),
])
def test_geometry_probe_delivers_image_for_each_provider(
        monkeypatch, tmp_path, model, expects_attachment):
    seen = {}
    image = tmp_path / 'geometry.png'

    def fake_render(*_args, **_kwargs):
        image.write_bytes(b'not-a-real-png')
        return str(image)

    def fake_run(prompt, **kwargs):
        seen.update(prompt=prompt, images=kwargs.get('images'))
        return {'text': 'OK', 'usage': {}, 'stdout': '', 'stderr': '',
                'event_types': {}, 'returncode': 0, 'argv': [],
                'provider': providers.provider_for(model), 'model': model,
                'effort': 'low', 'prompt_sha': 'x', 'n_images': int(expects_attachment)}

    monkeypatch.setattr(geometry, 'render_unique', fake_render)
    monkeypatch.setattr(geometry.P, 'run', fake_run)
    geometry.probe(224, 224, model, 1, 'low')
    if expects_attachment:
        assert seen['images'] == [str(image)]
        assert str(image) not in seen['prompt']
    else:
        assert seen['images'] is None
        assert str(image) in seen['prompt']


def test_protocol_decoders_reject_arbitrary_junk_and_excess_bits():
    assert decode_value('de-ad-be-ef', 'hex', 32) == 0xDEADBEEF
    assert decode_value('junkdeadbeef', 'hex', 32) is None
    value = 123456
    encoded = enc_b32(value, 64)
    assert decode_value(encoded.replace('0', 'O'), 'b32', 64) == value
    assert decode_value('Z' + encoded, 'b32', 64) is None
    assert ebind_dec_hex('0123-4567-89ab-cdef') == 0x0123456789ABCDEF
    assert ebind_dec_hex('junk0123456789abcdef') is None


@pytest.mark.parametrize('layout', [(8, 56), (32, 32)])
def test_structured_codeword_round_trip_and_wrong_block_rejection(layout):
    index_bits, tag_bits = layout
    corpus = build_corpus('unit-code', n_blocks=2, per_block=3)
    page, block = corpus['pages'][0]
    rec = corpus['blocks'][(page, block)][1]
    value = make_code(page, block, 1, rec, index_bits, tag_bits)
    got, why = verify_code(value, page, block, corpus, index_bits, tag_bits)
    assert why == 'ok' and got == rec
    other_page, other_block = corpus['pages'][1]
    assert verify_code(value, other_page, other_block, corpus, index_bits, tag_bits)[1] == 'tag_fail'


def test_no_answer_unreadable_is_not_correct_no_match():
    corpus = build_corpus('unit-outcomes', n_blocks=1, per_block=3)
    query = next(q for q in make_queries(corpus, 'unit-outcomes', 1, 1)
                 if q['kind'] == 'absent')
    assert classify({'result': 'NO_MATCH'}, query, corpus, 'hex')['outcome'] == 'N'
    assert classify({'result': 'UNREADABLE'}, query, corpus, 'hex')['outcome'] == 'P0'


def test_wrong_valid_handle_is_false_accept():
    corpus = build_corpus('unit-wrong-valid', n_blocks=1, per_block=3)
    page, block = corpus['pages'][0]
    query = {'kind': 'present', 'gold': (page, block, 0)}
    wrong = corpus['blocks'][(page, block)][1]
    code = enc_hex(make_code(page, block, 1, wrong))
    result = classify({'page': page, 'block': block, 'code': code}, query, corpus, 'hex')
    assert result['outcome'] == 'W' and result['validator'] == 'ok'


def test_arm_b_base32_reduces_native_archive_token_cost():
    corpus = build_corpus('unit-arm-b', n_blocks=1, per_block=12)
    page, block = corpus['pages'][0]
    _, _, _, hex_meta = render_block(corpus, page, block, 'hex', slot=0, patch=32)
    _, _, _, b32_meta = render_block(corpus, page, block, 'b32', slot=0, patch=32)
    hex_tokens = image_tokens(hex_meta['w'], hex_meta['h'], 32, 1.2)
    b32_tokens = image_tokens(b32_meta['w'], b32_meta['h'], 32, 1.2)
    assert b32_tokens < hex_tokens


def test_confirm_worker_receives_declared_effort(monkeypatch):
    seen = []
    def fake_one(job, model, effort):
        seen.append((job, model, effort))
        return [{'job': job}]
    monkeypatch.setattr(confirm, 'one', fake_one)
    rows = confirm.execute_jobs(['a', 'b'], 'gpt-5.6-sol', 'high', workers=1)
    assert rows == [{'job': 'a'}, {'job': 'b'}]
    assert seen == [('a', 'gpt-5.6-sol', 'high'), ('b', 'gpt-5.6-sol', 'high')]
