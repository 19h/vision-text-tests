#!/usr/bin/env python3
"""Fail-closed semantic verifier for fetched E-BIND canonical records.

This is deliberately separate from the gold scorer. It sees only a user query and one
candidate canonical record, exactly as a deployable post-fetch reconciliation step would.
Wrong-valid candidates exercise the failure that a keyed codeword cannot detect.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import time

from ebind1 import build_corpus, make_queries
import providers as P
import provenance as V

ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT = (
    "You are a fail-closed verifier for an archive lookup. Decide whether the fetched\n"
    "canonical record exactly satisfies the user's query. Every factual component must\n"
    "match: subject, action, quantity, and goods. Similar is not sufficient.\n\n"
    "QUERY: {query}\n"
    "FETCHED CANONICAL RECORD: {record}\n\n"
    'Return STRICT JSON only: {{"accept": true}} or {{"accept": false}}. '
    "If uncertain, return false."
)


def record_at(corpus, locator):
    page, block, index = locator
    return corpus['blocks'][(page, block)][index]


def wrong_valid_decoy(query, corpus):
    """Deterministic same-block decoy, preferring the closest semantic record."""
    gold = record_at(corpus, query['gold'])
    candidates = [r for r in corpus['blocks'][(gold['page'], gold['block'])]
                  if r['index'] != gold['index']]
    def score(r):
        shared = sum(r[k] == gold[k] for k in ('subj', 'verb', 'goods'))
        return (shared, -abs(r['qty'] - gold['qty']), -r['index'])
    return max(candidates, key=score)


def candidate_jobs(corpus, queries):
    jobs = []
    for query in queries:
        if query['kind'] == 'present':
            jobs.append((query, record_at(corpus, query['gold']), 'correct', True))
            jobs.append((query, wrong_valid_decoy(query, corpus), 'wrong_valid_decoy', False))
        else:
            jobs.append((query, record_at(corpus, query['injected_candidate']),
                         'no_answer_candidate', False))
    return jobs


def run_one(job, model, effort):
    query, candidate, candidate_kind, truth_matches = job
    prompt = PROMPT.format(query=query['q'], record=candidate['text'])
    started = time.time()
    response = P.run(prompt, model=model, effort=effort, timeout=300, cwd=ROOT)
    answer = P.parse_json_answer(response['text'])
    parsed_accept = answer.get('accept') if isinstance(answer, dict) else None
    parse_ok = isinstance(parsed_accept, bool)
    # Production policy is fail closed: malformed/uncertain output never accepts a record.
    runtime_accept = parsed_accept is True if parse_ok else False
    row = {
        'item_id': query['item_id'], 'query': query['q'], 'query_kind': query['kind'],
        'sampling_page': query['sampling_page'], 'sampling_block': query['sampling_block'],
        'candidate_kind': candidate_kind, 'candidate_id':
            f"{candidate['page']}/{candidate['block']}/{candidate['index']}",
        'candidate_text': candidate['text'], 'truth_matches': truth_matches,
        'parsed_accept': parsed_accept, 'parse_ok': parse_ok,
        'runtime_accept': runtime_accept, 'model': model,
        'provider': P.provider_for(model), 'effort': P.effective_effort(model, effort),
        'seconds': round(time.time() - started, 1),
        'response': V.response_record(response),
    }
    print(f"  {candidate_kind:20s} truth={truth_matches!s:5s} "
          f"accept={runtime_accept!s:5s} parse={parse_ok!s:5s} {row['seconds']}s")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--stage', default='heldout-v1')
    ap.add_argument('--present', type=int, default=100)
    ap.add_argument('--absent', type=int, default=20)
    ap.add_argument('--blocks', type=int, default=12)
    ap.add_argument('--per-block', type=int, default=12, dest='per_block')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--effort', default='low')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--allow-mutable-model-alias', action='store_true')
    a = ap.parse_args()
    if P.model_is_mutable_alias(a.model) and not a.allow_mutable_model_alias:
        ap.error('--model must be an exact immutable model id (or opt into an exploratory alias)')
    suffix = '.dry-run' if a.dry_run else ''
    path = os.path.join(ROOT, f'results_verifier_{a.stage}_{a.model}{suffix}.json')
    V.require_new_output(path, a.overwrite)
    corpus = build_corpus(a.stage, a.blocks, a.per_block)
    if a.present > len(corpus['records']):
        ap.error('--present exceeds the generated corpus size')
    queries = make_queries(corpus, a.stage, a.present, a.absent)
    jobs = candidate_jobs(corpus, queries)
    print(f"semantic verifier: {len(jobs)} calls over {len(queries)} unique queries")
    if a.dry_run:
        rows = []
        print('dry-run: candidates constructed; no model calls made')
    else:
        with ThreadPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(lambda job: run_one(job, a.model, a.effort), jobs))
    provider = P.provider_for(a.model)
    manifest = V.manifest(
        experiment='ebind-semantic-verifier-v1', model=a.model, provider=provider,
        effort=a.effort, cli_version=P.cli_version(provider), harness_path=__file__,
        prompts={'verifier': PROMPT}, stage=a.stage, present=a.present, absent=a.absent,
        blocks=a.blocks, per_block=a.per_block, fail_closed=True,
        candidates={'correct': a.present, 'wrong_valid_decoy': a.present,
                    'no_answer_candidate': a.absent}, dry_run=a.dry_run)
    V.dump_json(path, {'schema_version': V.RESULT_SCHEMA_VERSION,
                       'manifest': manifest, 'results': rows})
    print(f"{len(rows)} completed -> {path}")


if __name__ == '__main__':
    main()
