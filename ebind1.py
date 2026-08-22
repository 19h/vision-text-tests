#!/usr/bin/env python3
"""E-BIND-1: can a semantic query resolve the correct CANONICAL record?

Amendments applied to the first draft of this spec:
 1. one pinned model+endpoint for both fidelity and geometry (no Opus/Sonnet composition)
 2. gold scorer (knows R_gold) kept strictly separate from any deployable runtime verifier
 3. answer-present and no-answer queries use different outcome partitions
 4. repeated passes estimate call variance, NOT extra independent items
 5. encoding compared in two arms: fixed-slot (accuracy) vs native-density (goodput)
 6. structural IDs are exact text outside the bitmap; label->image binding measured explicitly
"""
import os, sys, json, re, math, random, hmac, hashlib, subprocess, time, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_lib import bitmap, canvas, draw_lines, tokens, geom, image_tokens
import providers as P
import provenance as V
ROOT = os.path.dirname(os.path.abspath(__file__))
B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
PROTO = "ebind1-v1"
ARCHIVE = "A1"
REVISION = "r0001"                     # binds handles to a canonical state generation
KEY = b'ebind1-preflight-key-not-secret'

SUBJ = ["Depot Kilo","Sector 14","Unit Bravo","Terminal 7","Crew Delta","Warehouse 3",
        "Convoy Echo","Platform 22","Berth 19","Node 41","The cold store","Line Charlie"]
VERB = ["shipped","rejected","rerouted","inspected","quarantined","weighed","impounded"]
GOODS= ["crates of tin","pallets of resin","drums of glycol","spools of copper",
        "bales of flax","canisters of argon","reels of fibre","tubs of pitch"]

# ─────────────────────────────────────────── codeword: 8-bit index + 56-bit keyed tag
def digest_record(rec):
    return hashlib.blake2b(rec['text'].encode(), digest_size=8).hexdigest()

def make_code(page, block, index, rec, index_bits=8, tag_bits=56):
    if index_bits + tag_bits != 64 or min(index_bits, tag_bits) <= 0:
        raise ValueError('codeword layout must contain two positive fields totalling 64 bits')
    layout = '' if (index_bits, tag_bits) == (8, 56) else f'|layout={index_bits}+{tag_bits}'
    msg = '|'.join([PROTO, ARCHIVE, REVISION, page, block, str(index), digest_record(rec)]) + layout
    tag = hmac.new(KEY, msg.encode(), hashlib.blake2b).digest()
    tag_value = int.from_bytes(tag, 'big') & ((1 << tag_bits) - 1)
    if not (0 <= index < (1 << index_bits)):
        raise ValueError(f'index {index} does not fit in {index_bits} bits')
    return (index << tag_bits) | tag_value

def verify_code(v, page, block, corpus, index_bits=8, tag_bits=56):
    """Deployable-style validator: range, canonical form, index range, keyed tag."""
    if v is None or not (0 <= v < (1 << 64)): return None, 'range'
    index = v >> tag_bits
    recs = corpus['blocks'].get((page, block))
    if recs is None: return None, 'no_such_block'
    if index >= len(recs): return None, 'bad_index'
    if make_code(page, block, index, recs[index], index_bits, tag_bits) != v:
        return None, 'tag_fail'
    return recs[index], 'ok'

def enc_hex(v): return '%016x' % v
def enc_b32(v):
    s = ''
    for _ in range(13): s = B32[v & 31] + s; v >>= 5
    return s
def dec_hex(s):
    t = re.sub(r'[\s\-]', '', str(s))
    return int(t, 16) if len(t) == 16 and re.fullmatch(r'[0-9a-fA-F]{16}', t) else None
def dec_b32(s):
    t = re.sub(r'[\s\-]', '', s).upper().replace('O', '0').replace('I', '1').replace('L', '1')
    if len(t) != 13 or any(c not in B32 for c in t): return None
    v = 0
    for c in t: v = v * 32 + B32.index(c)
    if v >= (1 << 64): return None                      # excess high bits -> reject
    if enc_b32(v) != t: return None                     # canonical re-encode check
    return v

# ─────────────────────────────────────────── corpus
def build_corpus(seed, n_blocks=6, per_block=12, decoy_rate=0.35):
    rng = random.Random(f'{PROTO}|{seed}')
    qty_pool = rng.sample(range(1000, 9999), n_blocks * per_block * 2)
    corpus = {'blocks': {}, 'records': [], 'pages': []}
    qi = 0
    for b in range(n_blocks):
        page = f"P{b//3 + 1:02d}"; block = f"B{b%3 + 1:02d}"
        corpus['pages'].append((page, block))
        recs = []
        base_pairs = []
        for i in range(per_block):
            if base_pairs and rng.random() < decoy_rate:
                s, g = rng.choice(base_pairs)              # near-semantic decoy
            else:
                s, g = rng.choice(SUBJ), rng.choice(GOODS); base_pairs.append((s, g))
            v = rng.choice(VERB); q = qty_pool[qi]; qi += 1
            rec = dict(page=page, block=block, index=i, subj=s, verb=v, qty=q, goods=g,
                       text=f"{s} {v} {q} {g}")
            recs.append(rec); corpus['records'].append(rec)
        corpus['blocks'][(page, block)] = recs
    return corpus

def make_queries(corpus, seed, n_present=20, n_absent=5):
    rng = random.Random(f'{PROTO}|q|{seed}')
    qs = []
    for r in rng.sample(corpus['records'], n_present):
        qs.append(dict(kind='present', q=f"which record says {r['subj']} {r['verb']} {r['qty']} {r['goods']}",
                       gold=(r['page'], r['block'], r['index']),
                       sampling_page=r['page'], sampling_block=r['block']))
    used = {r['qty'] for r in corpus['records']}
    for _ in range(n_absent):
        while True:
            q = rng.randint(1000, 9999)
            if q not in used: break
        r = rng.choice(corpus['records'])
        qs.append(dict(kind='absent', q=f"which record says {r['subj']} {r['verb']} {q} {r['goods']}",
                       gold=None, sampling_page=r['page'], sampling_block=r['block'],
                       injected_candidate=(r['page'], r['block'], r['index'])))
    rng.shuffle(qs)
    for q in qs:
        q['item_id'] = hashlib.sha256(
            f"{PROTO}|{seed}|{q['kind']}|{q['q']}".encode()).hexdigest()[:16]
    return qs

# ─────────────────────────────────────────── rendering
def render_block(corpus, page, block, enc, cell='6x13', slot=16, outdir=None, patch=28,
                 index_bits=8, tag_bits=56):
    """Arm A uses slot=16 for both encodings (base32 padded); Arm B uses native length."""
    f = bitmap(cell)
    m = re.search(r'(\d+)x(\d+)', cell)
    cw, lh = int(m.group(1)), int(m.group(2))
    recs = corpus['blocks'][(page, block)]
    lines = []
    for r in recs:
        v = make_code(page, block, r['index'], r, index_bits, tag_bits)
        code = enc_hex(v) if enc == 'hex' else enc_b32(v)
        code = code.ljust(slot) if slot else code
        lines.append(f"[{code}] {r['text']}")
    width = max(len(l) for l in lines) + 1
    uw = patch * cw // math.gcd(patch, cw)
    uh = patch * lh // math.gcd(patch, lh)
    w = uw * max(1, math.ceil(width * cw / uw))
    h = uh * max(1, math.ceil(len(lines) * lh / uh))
    im, d = canvas(w, h); draw_lines(d, f, lines, 0, 0, lh=lh)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, f"{enc}_{page}_{block}.png"); im.save(p)
        return p, im, lines, dict(w=w, h=h, records=len(lines), patch=patch,
                                  slot=slot, chars=sum(len(x) for x in lines))
    return None, im, lines, dict(w=w, h=h, records=len(lines), patch=patch,
                                 slot=slot, chars=sum(len(x) for x in lines))

# ─────────────────────────────────────────── prompts
HDR_PATH = ("You are searching an archive. Each block is identified by exact text, followed by the\n"
            "path of an image holding that block's records. Every record line in an image reads:\n"
            "  [CODE] description\n\n")
HDR_ATTACH = ("You are searching an archive. Each block is identified by exact text, followed by\n"
              "which of the attached images holds that block's records. Every record line in an\n"
              "image reads:\n  [CODE] description\n\n")
TAIL = ("\nQuestion: {q}\n\n"
        "Find the ONE record matching the question. Open only the blocks you need.\n"
        "Return STRICT JSON, nothing else, one of:\n"
        '  {{"page": "Pxx", "block": "Bxx", "code": "<the record\'s CODE exactly>"}}\n'
        '  {{"result": "NO_MATCH"}}   if no record in the archive matches\n'
        '  {{"result": "UNREADABLE"}} if you found it but cannot read its code\n'
        "Do not guess a code. Answer from the images alone; do not run any command to crop,\n"
        "zoom or enhance them.")

def build_prompt(paths, q, provider='claude'):
    """codex receives the images as attachments, so blocks are addressed by attachment
    order rather than by path - the model never sees a filesystem path."""
    items = list(paths.items())
    if provider == 'codex':
        body = ''.join(f"ARCHIVE {ARCHIVE} - PAGE {p} - BLOCK {b}: attached image {i+1}\n"
                       for i, ((p, b), _) in enumerate(items))
        return HDR_ATTACH + body + TAIL.format(q=q)
    body = ''.join(f"ARCHIVE {ARCHIVE} - PAGE {p} - BLOCK {b}: {path}\n" for (p, b), path in items)
    return HDR_PATH + body + TAIL.format(q=q)

def ordered_images(paths):
    return [path for _, path in paths.items()]

def build_text_prompt(corpus, q, index_bits=8, tag_bits=56):
    """Semantic ceiling control: identical corpus and queries, exact text carrier."""
    body = ''
    for (p, b), recs in corpus['blocks'].items():
        body += f"ARCHIVE {ARCHIVE} - PAGE {p} - BLOCK {b}\n"
        for r in recs:
            body += f"  [{enc_hex(make_code(p, b, r['index'], r, index_bits, tag_bits))}] {r['text']}\n"
    return HDR_PATH + body + TAIL.format(q=q)

# ─────────────────────────────────────────── run + gold scoring
def classify(ans, query, corpus, enc, index_bits=8, tag_bits=56):
    """GOLD scorer - it knows R_gold. Deliberately NOT a deployable runtime verifier."""
    present = query['kind'] == 'present'
    out = dict(kind=query['kind'], returned_page=ans.get('page'), returned_block=ans.get('block'),
               returned_code=ans.get('code'), result=ans.get('result'))
    if ans.get('_parse_fail'): out['outcome'] = 'P' if present else 'P0'; return out
    if ans.get('result') == 'NO_MATCH':
        out['outcome'] = 'A' if present else 'N'; return out
    if ans.get('result') == 'UNREADABLE':
        # UNREADABLE is a valid abstention for answer-present items, but it does not
        # establish absence. The no-answer partition has no abstention-success bucket.
        out['outcome'] = 'A' if present else 'P0'; return out
    pg, bl, code = ans.get('page'), ans.get('block'), ans.get('code') or ''
    if not pg or not bl or not code:
        out['outcome'] = 'P' if present else 'P0'; return out
    v = dec_hex(code) if enc == 'hex' else dec_b32(code)
    rec, why = verify_code(v, pg, bl, corpus, index_bits, tag_bits)
    out['validator'] = why
    if rec is None:
        out['outcome'] = 'D' if present else 'D0'; return out
    if present:
        g = query['gold']
        out['correct_block'] = (pg, bl) == (g[0], g[1])
        out['outcome'] = 'C' if (pg, bl, rec['index']) == g else 'W'
    else:
        out['outcome'] = 'W0'
    return out

def call(prompt, model, images=None, effort='low', timeout=900):
    prov = P.provider_for(model)
    r = P.run(prompt, model=model, images=P.images_for(prov, images or []),
              effort=effort, timeout=timeout, cwd=ROOT)
    ans = P.parse_json_answer(r['text'])
    raw = V.response_record(r)
    if not isinstance(ans, dict): return {'_parse_fail': True}, raw
    return ans, raw

def run_one(job):
    corpus, paths, query, enc, model, rep, carrier, effort, arm, index_bits, tag_bits = job
    t0 = time.time()
    prov = P.provider_for(model)
    if carrier == 'text':
        prompt, imgs = build_text_prompt(corpus, query['q'], index_bits, tag_bits), None
    else:
        prompt, imgs = build_prompt(paths, query['q'], prov), ordered_images(paths)
    ans, raw = call(prompt, model, images=imgs, effort=effort)
    res = classify(ans, query, corpus, 'hex' if carrier == 'text' else enc,
                   index_bits, tag_bits)
    res.update(enc=enc, carrier=carrier, arm=arm, rep=rep, model=model,
               provider=prov, effort=P.effective_effort(model, effort),
               codeword_layout=f'{index_bits}+{tag_bits}',
               query=query['q'], item_id=query['item_id'],
               sampling_page=query['sampling_page'], sampling_block=query['sampling_block'],
               gold=query['gold'], seconds=round(time.time() - t0), response=raw)
    print(f"  {carrier:5s} {enc:4s} rep{rep} {query['kind']:7s} -> {res['outcome']:2s} "
          f"({res.get('validator','-')}) {res['seconds']}s")
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='pin an exact model id')
    ap.add_argument('--stage', default='preflight')
    ap.add_argument('--present', type=int, default=20); ap.add_argument('--absent', type=int, default=5)
    ap.add_argument('--reps', type=int, default=1,
                    help='image-carrier passes over each frozen item')
    ap.add_argument('--text-reps', type=int, default=1,
                    help='text-ceiling passes; one is normally sufficient')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--arm', default='A', choices=['A', 'B'])
    ap.add_argument('--effort', default='low')
    ap.add_argument('--carriers', nargs='*', default=['image', 'text'])
    ap.add_argument('--encs', nargs='*', default=['hex', 'b32'])
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--per-block', type=int, default=12, dest='per_block')
    ap.add_argument('--patch', type=int, default=None,
                    help='rendering grid; Arm A defaults to 28, Arm B to the model-native grid')
    ap.add_argument('--index-bits', type=int, default=8)
    ap.add_argument('--tag-bits', type=int, default=56)
    ap.add_argument('--dry-run', action='store_true',
                    help='render and validate the frozen campaign without making model calls')
    ap.add_argument('--overwrite', action='store_true',
                    help='replace an existing result artifact (never implied by --dry-run)')
    ap.add_argument('--allow-mutable-model-alias', action='store_true',
                    help='exploratory compatibility only; frozen runs require an exact id')
    a = ap.parse_args()
    if P.model_is_mutable_alias(a.model) and not a.allow_mutable_model_alias:
        ap.error('--model must be an exact immutable model id (or explicitly opt into an exploratory alias)')
    if a.index_bits + a.tag_bits != 64 or min(a.index_bits, a.tag_bits) <= 0:
        ap.error('--index-bits and --tag-bits must be positive and sum to 64')
    corpus = build_corpus(a.stage, n_blocks=a.blocks, per_block=a.per_block)
    if a.present > len(corpus['records']):
        raise SystemExit(f"--present {a.present} exceeds corpus size "
                         f"{len(corpus['records'])} ({a.blocks} blocks x {a.per_block}); "
                         f"raise --blocks/--per-block")
    queries = make_queries(corpus, a.stage, a.present, a.absent)
    suffix = '.dry-run' if a.dry_run else ''
    fn = f'results_ebind1_{a.stage}_{a.arm}_{a.model}{suffix}.json'
    result_path = os.path.join(ROOT, fn)
    V.require_new_output(result_path, a.overwrite)
    slot = 16 if a.arm == 'A' else 0
    patch = a.patch or (28 if a.arm == 'A' else geom(a.model)['patch'])
    tokens_per_patch = 1.2 if patch == 32 else 1.0
    outdir = os.path.join(ROOT, 'images', f'EBIND1_{a.stage}_{a.arm}')
    paths, render_meta = {}, {}
    for enc in ('hex', 'b32'):
        paths[enc] = {}
        for (p, b) in corpus['pages']:
            fp, _, _, meta = render_block(corpus, p, b, enc, slot=slot,
                                          outdir=outdir, patch=patch,
                                          index_bits=a.index_bits, tag_bits=a.tag_bits)
            paths[enc][(p, b)] = fp
            render_meta[(enc, p, b)] = meta
    jobs = []
    for carrier in a.carriers:
        encs = ['hex'] if carrier == 'text' else a.encs
        for enc in encs:
            carrier_reps = a.text_reps if carrier == 'text' else a.reps
            for rep in range(1, carrier_reps + 1):
                for q in queries:
                    jobs.append((corpus, paths.get(enc, {}), q, enc, a.model, rep,
                                 carrier, a.effort, a.arm, a.index_bits, a.tag_bits))
    print(f"E-BIND-1 {a.stage} arm {a.arm}: {len(jobs)} calls, model={a.model}, "
          f"{len(corpus['records'])} records in {len(corpus['pages'])} blocks")
    if a.dry_run:
        out = []
        print('dry-run: rendered campaign; no model calls made')
    else:
        with ThreadPoolExecutor(a.jobs) as ex:
            out = list(ex.map(run_one, jobs))
    prov = P.provider_for(a.model)
    archive_geometry = {}
    for enc in ('hex', 'b32'):
        ms = [meta for (e, _, _), meta in render_meta.items() if e == enc]
        total_tokens = sum(image_tokens(m['w'], m['h'], patch, tokens_per_patch) for m in ms)
        archive_geometry[enc] = {
            'images': len(ms), 'records': len(corpus['records']),
            'image_tokens': total_tokens,
            'records_per_image_token': len(corpus['records']) / total_tokens,
            'canvases': [f"{m['w']}x{m['h']}" for m in ms],
        }
    all_images = [path for enc_paths in paths.values() for path in enc_paths.values()]
    run_manifest = V.manifest(
        experiment=PROTO, model=a.model, provider=prov, effort=a.effort,
        cli_version=P.cli_version(prov), harness_path=__file__,
        prompts={'image_template': HDR_PATH + TAIL, 'attachment_template': HDR_ATTACH + TAIL},
        images=all_images,
        stage=a.stage, arm=a.arm, protocol=PROTO, carriers=a.carriers, encodings=a.encs,
        present=a.present, absent=a.absent, image_reps=a.reps, text_reps=a.text_reps,
        blocks=a.blocks, patch=patch, tokens_per_patch=tokens_per_patch,
        per_block=a.per_block, n_records=len(corpus['records']),
        codeword_layout={'index_bits': a.index_bits, 'tag_bits': a.tag_bits},
        mutable_model_alias=P.model_is_mutable_alias(a.model),
        archive_geometry=archive_geometry, dry_run=a.dry_run)
    V.dump_json(result_path,
                dict(schema_version=V.RESULT_SCHEMA_VERSION, manifest=run_manifest,
                     model=a.model, provider=prov,
                     effort=P.effective_effort(a.model, a.effort),
                     cli=run_manifest['cli_version'], stage=a.stage, arm=a.arm,
                     protocol=PROTO, n_records=len(corpus['records']), results=out))
    import collections
    for carrier in a.carriers:
        for enc in (['hex'] if carrier == 'text' else a.encs):
            rs = [r for r in out if r['carrier'] == carrier and r['enc'] == enc]
            if not rs: continue
            c = collections.Counter(r['outcome'] for r in rs)
            pres = [r for r in rs if r['kind'] == 'present']
            print(f"\n{carrier}/{enc}: " + '  '.join(f"{k}={v}" for k, v in sorted(c.items())))
            if pres:
                print(f"  P(correct block) = {sum(1 for r in pres if r.get('correct_block'))}/{len(pres)}")
    print(f"\n-> {fn}")
