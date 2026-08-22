from analyze_ebind import paired_carrier_difference, paired_condition_difference, summarize
from analyze_encoding import compare as compare_encoding
from campaign import plan, stage_complete
from layout_probe import render
from delim_probe import render as render_delimited
from analyze_verifier import analyze as analyze_verifier
from ebind1 import build_corpus, make_queries
from verifier_probe import candidate_jobs


def row(item, page, rep, carrier, kind, outcome):
    return {'item_id': item, 'sampling_page': page, 'rep': rep, 'carrier': carrier,
            'enc': 'hex', 'kind': kind, 'outcome': outcome, 'query': item}


def test_repeated_calls_are_grouped_as_items():
    rows = [row('a', 'P1', rep, 'image', 'present', outcome)
            for rep, outcome in enumerate(('C', 'C', 'W'), 1)]
    rows += [row('b', 'P2', rep, 'image', 'present', 'C') for rep in (1, 2, 3)]
    report = summarize(rows, bootstrap_reps=100)
    assert report['image/hex']['items'] == 2
    assert report['image/hex']['calls'] == 6
    assert report['image/hex']['present']['W']['rate'] == 1 / 6
    assert report['image/hex']['present']['W']['cluster'] == 'page_then_item'


def test_ebind_descriptive_metrics_include_latency_retry_usage_and_binding():
    rows = [dict(row('a', 'P1', 1, 'image', 'present', 'C'), seconds=2,
                 correct_block=True, validator='ok',
                 response={'usage': {'input_tokens': 100, 'output_tokens': 5}}),
            dict(row('b', 'P2', 1, 'image', 'present', 'W'), seconds=4,
                 correct_block=False, validator='ok',
                 response={'usage': {'input_tokens': 100, 'output_tokens': 5}})]
    metrics = summarize(rows, bootstrap_reps=100)['image/hex']['descriptive']
    assert metrics['latency_seconds'] == {'n': 2, 'p50': 3.0, 'p95': 3.9}
    assert metrics['geometric_attempts_if_iid']['present_correct'] == 2
    assert metrics['usage_tokens']['input_total'] == 200
    assert metrics['correct_block_rate_present'] == .5
    assert metrics['correct_record_given_correct_block'] == 1


def test_paired_carrier_contrast_uses_unique_items():
    rows = [row('a', 'P1', 1, 'image', 'present', 'C'),
            row('a', 'P1', 2, 'image', 'present', 'W'),
            row('a', 'P1', 1, 'text', 'present', 'C')]
    result = paired_carrier_difference(rows, 'present', 'C', reps=100)
    assert result['n_paired_items'] == 1
    assert result['difference'] == -0.5


def test_paired_arm_contrast_clusters_by_page_and_item():
    rows = []
    for item, page, a_out, b_out in [('a', 'P1', 'C', 'W'), ('b', 'P2', 'C', 'C')]:
        rows += [dict(row(item, page, 1, 'image', 'present', a_out), arm='A'),
                 dict(row(item, page, 1, 'image', 'present', b_out), arm='B')]
    result = paired_condition_difference(
        rows, 'present', 'C', lambda r: r['arm'], 'A', 'B', reps=100)
    assert result['difference'] == .5
    assert result['n_paired_items'] == 2
    assert result['cluster'] == 'page_then_item'


def test_layout_conditions_share_underlying_values():
    values = []
    for grouped in (False, True):
        for delimited in (False, True):
            _, _, v, _, _ = render('6x13', 128, 7, grouped, delimited)
            values.append(v)
    assert all(v == values[0] for v in values[1:])


def test_delimited_encodings_are_paired_on_values_canvas_and_target_position():
    rendered = [render_delimited('6x13', 6, 13, enc, 128, 'r7')
                for enc in ('hex', 'b32')]
    _, hex_lines, hex_values, hex_w, hex_h, _, _ = rendered[0]
    _, b32_lines, b32_values, b32_w, b32_h, _, _ = rendered[1]
    assert hex_values == b32_values
    assert (hex_w, hex_h) == (b32_w, b32_h)
    assert hex_lines[0].index('[') == b32_lines[0].index('[')
    first_values = [render_delimited('6x13', 6, 13, 'hex', 128, f'r{rep}')[2][0]
                    for rep in range(1, 6)]
    assert len(set(first_values)) == len(first_values)


def test_campaign_pairs_arm_a_and_b_on_same_heldout_namespace():
    stages = {s['name']: s for s in plan('gpt-5.6-sol')}
    a = stages['heldout_stage1_structured_32_32']['command']
    b = stages['arm_b_native_goodput']['command']
    assert a[a.index('--stage') + 1] == b[b.index('--stage') + 1] == 'heldout-v1'
    assert '--index-bits' in a and a[a.index('--index-bits') + 1] == '32'
    assert sum(s['estimated_calls'] for s in stages.values()) == 2265
    all_outputs = [name for stage in stages.values() for name in stage['outputs']]
    assert all(stage['outputs'] for stage in stages.values())
    assert len(all_outputs) == len(set(all_outputs))


def test_campaign_resume_requires_complete_non_dry_result(tmp_path):
    stage = {'estimated_calls': 2, 'outputs': ['results_unit.json']}
    result = {'schema_version': 'vision-text-result-v2',
              'manifest': {'dry_run': False}, 'results': [{}, {}]}
    (tmp_path / 'results_unit.json').write_text(__import__('json').dumps(result))
    assert stage_complete(stage, str(tmp_path))
    result['manifest']['dry_run'] = True
    (tmp_path / 'results_unit.json').write_text(__import__('json').dumps(result))
    assert not stage_complete(stage, str(tmp_path))
    result['manifest']['dry_run'] = False
    result['results'].pop()
    (tmp_path / 'results_unit.json').write_text(__import__('json').dumps(result))
    assert not stage_complete(stage, str(tmp_path))


def test_verifier_campaign_exercises_correct_and_wrong_valid_candidates():
    corpus = build_corpus('heldout-v1', n_blocks=12, per_block=12)
    queries = make_queries(corpus, 'heldout-v1', 100, 20)
    jobs = candidate_jobs(corpus, queries)
    assert len(jobs) == 220
    assert sum(kind == 'correct' and truth for _, _, kind, truth in jobs) == 100
    assert sum(kind == 'wrong_valid_decoy' and not truth for _, _, kind, truth in jobs) == 100
    assert sum(kind == 'no_answer_candidate' and not truth for _, _, kind, truth in jobs) == 20
    for query, candidate, kind, _ in jobs:
        if kind == 'wrong_valid_decoy':
            assert (candidate['page'], candidate['block'], candidate['index']) != query['gold']


def test_verifier_analysis_is_fail_closed_and_paired():
    rows = []
    for i in range(8):
        base = {'item_id': f'i{i}', 'sampling_page': f'P{i % 2}', 'parse_ok': True}
        rows.append(dict(base, candidate_kind='correct', truth_matches=True,
                         runtime_accept=True))
        rows.append(dict(base, candidate_kind='wrong_valid_decoy', truth_matches=False,
                         runtime_accept=False))
    report = analyze_verifier(rows, reps=100)
    assert report['true_accept']['rate'] == 1
    assert report['false_accept_all_wrong']['rate'] == 0
    assert report['false_accept_all_wrong']['ci95'][1] > 0
    assert report['paired_discrimination']['rate'] == 1


def test_encoding_equivalence_requires_interval_inside_margin():
    rows = []
    for rep in range(1, 21):
        rows += [{'bits': 64, 'rep': rep, 'enc': 'hex', 'value': True},
                 {'bits': 64, 'rep': rep, 'enc': 'b32', 'value': True}]
    result = compare_encoding(rows, 64, margin=.10, reps=200)
    assert result['n'] == 20 and result['difference'] == 0
    assert not result['equivalent']
    rows = []
    for rep in range(1, 187):
        rows += [{'bits': 64, 'rep': rep, 'enc': 'hex', 'value': True},
                 {'bits': 64, 'rep': rep, 'enc': 'b32', 'value': True}]
    assert compare_encoding(rows, 64, margin=.10, reps=200)['equivalent']


def test_cluster_interval_is_not_zero_width_at_boundary():
    rows = [row(f'i{i}', f'P{i % 4}', 1, 'image', 'present', 'C') for i in range(20)]
    interval = summarize(rows, bootstrap_reps=100)['image/hex']['present']['W']
    assert interval['rate'] == 0
    assert interval['ci95'][1] > 0
    assert interval['cluster'].endswith('wilson_boundary')
