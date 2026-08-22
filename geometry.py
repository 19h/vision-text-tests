#!/usr/bin/env python3
"""Provider-agnostic image-geometry probe.

Measures px-per-image-token and locates the provider's downscale ceiling.

Method notes, all learned the hard way on the Claude side:
  * Do NOT assume a constant prompt overhead - it is not constant.
  * Do NOT reuse an image across reps: on a repeat call it lands in `cached_input_tokens`
    and a naive metric collapses to ~0. Render UNIQUE content for every single call.
  * Compare sizes by delta, not against an assumed baseline.
"""
import os, sys, json, math, random, argparse, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_lib import bitmap, canvas, draw_lines, numbered_lines
import providers as P
import provenance as V
ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT = "Reply with only the word OK."

def render_unique(w, h, tag, cell='6x10'):
    f = bitmap(cell)
    cw, lh = (int(x) for x in cell.split('x'))
    rng = random.Random(f'geom|{w}x{h}|{tag}')
    lines, _ = numbered_lines(rng, w // cw, h // lh)
    im, d = canvas(w, h); draw_lines(d, f, lines, 0, 0)
    p = os.path.join('/tmp', f'geom_{w}x{h}_{tag}.png'); im.save(p)
    return p

def probe(w, h, model, rep, effort, timeout=300):
    img = render_unique(w, h, f'r{rep}')
    image_sha = V.sha256_file(img)
    provider = P.provider_for(model)
    prompt = P.image_head(provider, img, desc="is a geometry-accounting stimulus") + PROMPT
    r = P.run(prompt, model=model, images=P.images_for(provider, [img]),
              effort=effort, timeout=timeout)
    try: os.remove(img)
    except OSError: pass
    u = r['usage'] or {}
    return dict(w=w, h=h, rep=rep, model=model,
                effort=P.effective_effort(model, effort),
                input_tokens=u.get('input_tokens'), cached=u.get('cached_input_tokens'),
                cache_write=u.get('cache_write_input_tokens'),
                output_tokens=u.get('output_tokens'), ok=r['text'] is not None,
                text=(r['text'] or '')[:40], image_sha256=image_sha,
                response=V.response_record(r))

def analyse(rows, patch):
    by = {}
    for r in rows:
        if r['input_tokens']: by.setdefault((r['w'], r['h']), []).append(r['input_tokens'])
    keys = sorted(by, key=lambda k: k[0] * k[1])
    print(f"\n{'canvas':>12s}{'pixels':>10s}{'patches':>9s}{'n':>3s}{'min':>8s}{'median':>8s}"
          f"{'d(px)/d(tok)':>14s}")
    prev = None
    out = []
    for k in keys:
        w, h = k; v = sorted(by[k]); px = w * h
        patches = math.ceil(w / patch) * math.ceil(h / patch)
        marg = ''
        if prev:
            dpx = px - prev[0]; dtok = min(v) - prev[1]
            marg = f"{dpx/dtok:.0f}" if dtok > 0 else 'n/a'
        out.append(dict(w=w, h=h, px=px, patches=patches, n=len(v), min=min(v),
                        median=st.median(v), marginal=marg))
        print(f"{f'{w}x{h}':>12s}{px:>10d}{patches:>9d}{len(v):>3d}{min(v):>8d}"
              f"{st.median(v):>8.0f}{marg:>14s}")
        prev = (px, min(v))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--sizes', nargs='+', type=int, default=[224, 448, 896, 1344])
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--patch', type=int, default=32)
    ap.add_argument('--effort', default='low')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--tag', default='rate')
    ap.add_argument('--rectangles', nargs='*', default=[], metavar='WxH',
                    help='additional non-square canvases, e.g. 1344x896 896x1600')
    ap.add_argument('--allow-mutable-model-alias', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if P.model_is_mutable_alias(a.model) and not a.allow_mutable_model_alias:
        ap.error('--model must be an exact immutable model id (or opt into an exploratory alias)')
    fn = f"results_geometry_{a.tag}_{a.model}.json"
    result_path = os.path.join(ROOT, fn)
    V.require_new_output(result_path, a.overwrite)
    from concurrent.futures import ThreadPoolExecutor
    rects = []
    for spec in a.rectangles:
        m = __import__('re').fullmatch(r'(\d+)[xX](\d+)', spec)
        if not m: ap.error(f'invalid rectangle {spec!r}; expected WxH')
        rects.append((int(m.group(1)), int(m.group(2))))
    canvases = [(s, s) for s in a.sizes] + rects
    jobs = [(w, h, rep) for rep in range(1, a.reps + 1) for w, h in canvases]
    print(f"geometry/{a.tag}: {len(jobs)} calls, model={a.model}, effort={a.effort}, "
          f"patch={a.patch}, cli={P.cli_version(P.provider_for(a.model))}")
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda j: probe(j[0], j[1], a.model, j[2], a.effort), jobs))
    for r in rows:
        print(f"  {r['w']}x{r['h']} rep{r['rep']}: input={r['input_tokens']} "
              f"cached={r['cached']} ok={r['ok']}")
    summary = analyse(rows, a.patch)
    prov = P.provider_for(a.model)
    run_manifest = V.manifest(
        experiment=f'geometry-{a.tag}-v2', model=a.model, provider=prov, effort=a.effort,
        cli_version=P.cli_version(prov), harness_path=__file__,
        prompts={'geometry_instruction': PROMPT,
                 'delivery_template': P.image_head(prov, '<IMAGE>',
                                                   desc='is a geometry-accounting stimulus')},
        patch=a.patch, reps=a.reps, canvases=[f'{w}x{h}' for w, h in canvases])
    V.dump_json(result_path,
                dict(schema_version=V.RESULT_SCHEMA_VERSION, manifest=run_manifest,
                     model=a.model, effort=P.effective_effort(a.model, a.effort), patch=a.patch,
                     cli=run_manifest['cli_version'], results=rows, summary=summary))
    print(f"-> {fn}")
