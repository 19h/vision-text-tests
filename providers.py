#!/usr/bin/env python3
"""Provider adapter: one call interface over `claude -p` and `codex exec`.

The two CLIs differ in a way that matters for this benchmark:

  claude -p : the image is referenced by PATH in the prompt and fetched by the Read tool.
              Other tools must be explicitly disallowed or the model will shell out and
              crop/zoom the image, which measures tooling rather than vision.
  codex exec: the image is ATTACHED to the prompt with -i. The model never receives a
              path, so that confound is structurally impossible - provided the prompt
              text never mentions a path.

Both are recorded with exact model slugs; never store a mutable alias.
"""
import json, os, subprocess, hashlib, shutil

CODEX_MODELS  = {'gpt-5.6-sol', 'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.5',
                 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.3-codex-spark'}
CLAUDE_TOOLS_OFF = 'Bash,Write,Edit,Glob,Grep,Task,WebFetch,WebSearch,NotebookEdit'
MUTABLE_MODEL_ALIASES = {'opus', 'sonnet', 'haiku', 'sol', 'luna', 'terra', 'latest'}

def provider_for(model):
    return 'codex' if model in CODEX_MODELS or model.startswith('gpt-') else 'claude'

def model_is_mutable_alias(model):
    return str(model).lower() in MUTABLE_MODEL_ALIASES

def effective_effort(model, requested):
    """Effort is controllable on Codex; Claude CLI runs are explicitly uncontrolled."""
    return requested if provider_for(model) == 'codex' else None

def cli_version(provider):
    exe = 'codex' if provider == 'codex' else 'claude'
    if not shutil.which(exe): return None
    try:
        return subprocess.run([exe, '--version'], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────── codex
def _codex(prompt, images, model, effort, schema_path, timeout, cwd):
    cmd = ['codex', 'exec', '--ephemeral', '--ignore-user-config', '--skip-git-repo-check',
           '-s', 'read-only', '-m', model, '-c', f'model_reasoning_effort={effort}', '--json']
    if schema_path: cmd += ['--output-schema', schema_path]
    for im in images or []: cmd += ['-i', im]
    # `-i` is variadic, so a positional prompt after it is swallowed as another image.
    # Pass the prompt on stdin instead - also avoids argv limits on long prompts.
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd,
                       input=prompt)
    text, usage, types = None, {}, {}
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if not ln.startswith('{'): continue
        try: ev = json.loads(ln)
        except json.JSONDecodeError: continue
        pay = ev.get('payload') if isinstance(ev.get('payload'), dict) else ev
        t = pay.get('type') or ev.get('type')
        types[t] = types.get(t, 0) + 1
        # `--json` stream reports usage on turn.completed; session files use token_count.
        if t == 'turn.completed' and isinstance(pay.get('usage'), dict):
            usage = pay['usage']
        if t == 'token_count':
            info = pay.get('info') or {}
            usage = info.get('last_token_usage') or info.get('total_token_usage') or usage
        if t in ('agent_message', 'assistant_message', 'message'):
            text = pay.get('message') or pay.get('text') or text
        if t == 'item.completed':
            it = pay.get('item') or {}
            if it.get('type') in ('agent_message', 'assistant_message'):
                text = it.get('text') or it.get('message') or text
    return dict(text=text, usage=usage, stdout=r.stdout, stderr=r.stderr,
                event_types=types, returncode=r.returncode, argv=cmd)

# ─────────────────────────────────────────────────────────────── claude
def _claude(prompt, images, model, effort, schema_path, timeout, cwd):
    cmd = ['claude', '-p', '--output-format', 'json', '--allowedTools', 'Read',
           '--disallowedTools', CLAUDE_TOOLS_OFF, '--model', model, prompt]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    text, usage = None, {}
    try:
        d = json.loads(r.stdout)
        text = d.get('result')
        u = d.get('usage', {})
        usage = dict(input_tokens=u.get('input_tokens', 0),
                     cached_input_tokens=u.get('cache_read_input_tokens', 0),
                     cache_write_input_tokens=u.get('cache_creation_input_tokens', 0),
                     output_tokens=u.get('output_tokens', 0))
    except json.JSONDecodeError:
        text = r.stdout.strip()
    return dict(text=text, usage=usage, stdout=r.stdout, stderr=r.stderr,
                event_types={}, returncode=r.returncode, argv=cmd[:-1])

def run(prompt, model, images=None, effort='low', schema_path=None, timeout=900, cwd=None):
    """Returns dict(text, usage, provider, model, effort, argv, ...). Never raises on a
    model-side failure; inspect `text is None` and `stderr`."""
    prov = provider_for(model)
    fn = _codex if prov == 'codex' else _claude
    try:
        out = fn(prompt, images, model, effort, schema_path, timeout, cwd or os.getcwd())
    except subprocess.TimeoutExpired as e:
        as_text = lambda value: (value.decode(errors='replace') if isinstance(value, bytes)
                                 else (value or ''))
        out = dict(text=None, usage={}, stdout=as_text(e.stdout), stderr=as_text(e.stderr),
                   event_types={}, returncode=None, argv=e.cmd, error='timeout')
    except FileNotFoundError as e:
        out = dict(text=None, usage={}, stdout='', stderr=str(e), event_types={},
                   returncode=None, argv=[], error='cli_not_found')
    out.update(provider=prov, model=model, effort=effort if prov == 'codex' else None,
               prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:16],
               n_images=len(images or []))
    return out

def parse_json_answer(text):
    """Last decodable JSON object in a response, including nested objects."""
    if not text: return None
    decoder = json.JSONDecoder()
    found = []
    i = 0
    while i < len(text):
        i = text.find('{', i)
        if i < 0: break
        try:
            value, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(value, dict):
            found.append(value)
            i += end
        else:
            i += 1
    return found[-1] if found else None


def image_head(provider, path, desc="contains dense small text"):
    """Provider-aware image-delivery sentence. The QUESTION that follows must stay
    byte-identical across providers; only this line may differ."""
    if provider == 'codex':
        return f"The image attached to this message {desc}.\n\n"
    return f"Read the image at {path}\n\n"

def images_for(provider, paths):
    """codex attaches images; claude is given paths in the prompt and uses Read."""
    return list(paths) if provider == 'codex' else None
