#!/usr/bin/env python3
"""Shared, versioned provenance records for benchmark artifacts and model runs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

RESULT_SCHEMA_VERSION = "vision-text-result-v2"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_sha(path: str) -> str:
    return sha256_file(path)[:16]


def prompt_sha(prompt: str) -> str:
    return sha256_bytes(prompt.encode("utf-8"))[:16]


def image_records(paths) -> list[dict]:
    return [
        {"path": os.path.relpath(os.path.abspath(p), os.getcwd()),
         "sha256": sha256_file(p)}
        for p in (paths or [])
    ]


def environment_record() -> dict:
    versions = {}
    for module in ("PIL", "numpy", "scipy"):
        try:
            mod = __import__(module)
            versions[module] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[module] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": versions,
    }


def manifest(*, experiment: str, model: str, provider: str, effort, cli_version,
             harness_path: str, prompts: dict[str, str], images=None, **extra) -> dict:
    requested_effort = effort
    effective_effort = effort if provider == 'codex' else None
    out = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider,
        "effort": effective_effort,
        "cli_version": cli_version,
        "harness_sha256": source_sha(harness_path),
        "prompts": {name: {"sha256": prompt_sha(text), "text": text}
                    for name, text in sorted(prompts.items())},
        "images": image_records(images),
        "environment": environment_record(),
    }
    if provider != 'codex' and requested_effort is not None:
        out["requested_effort_ignored"] = requested_effort
    out.update(extra)
    return out


def response_record(result: dict) -> dict:
    """Full provider response needed for deterministic audit/regrading.

    argv intentionally excludes prompt text in the Claude adapter. No secrets or ambient
    environment variables are captured.
    """
    return {
        "text": result.get("text"),
        "usage": result.get("usage") or {},
        "returncode": result.get("returncode"),
        "stderr": result.get("stderr", ""),
        "stdout": result.get("stdout", ""),
        "event_types": result.get("event_types") or {},
        "argv": result.get("argv") or [],
        "prompt_sha": result.get("prompt_sha"),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "effort": result.get("effort"),
        "n_images": result.get("n_images"),
        "error": result.get("error"),
    }


def dump_json(path: str, value) -> None:
    with open(path, "w") as f:
        json.dump(value, f, indent=1, sort_keys=True)
        f.write("\n")


def require_new_output(path: str, overwrite: bool = False) -> None:
    """Refuse to destroy a completed artifact after paid work unless explicitly allowed."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"output already exists: {path}; choose a new tag/path or pass --overwrite")


def result_rows(value) -> list:
    """Rows from either a legacy list or a v2 result envelope."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return value["results"]
    return []


def load_result_rows(path: str) -> list:
    with open(path) as f:
        return result_rows(json.load(f))
