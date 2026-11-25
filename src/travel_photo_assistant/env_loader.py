"""Minimal `.env` loader built on the standard library only.

The project deliberately avoids third-party SDK lock-in, so this reads the same
`.env` format that `python-dotenv` uses without adding a dependency.

Precedence rules:
1. Real environment variables always win over `.env` entries.
2. The first `.env` found wins (project root, then the current directory).
3. Malformed lines are ignored rather than raising, so a typo in `.env` never
   takes down the whole service.
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse_env_text(text: str) -> dict[str, str]:
    """Parse `KEY=VALUE` lines. Supports comments, blank lines and `export ` prefixes."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def find_env_file(start: Path | None = None) -> Path | None:
    """Locate the `.env` file: project root first, then the current working directory."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    # src/travel_photo_assistant/env_loader.py -> project root
    candidates.append(Path(__file__).resolve().parents[2] / ".env")
    candidates.append(Path.cwd() / ".env")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load `.env` into `os.environ`.

    Returns the mapping that was applied. Existing environment variables are
    preserved unless `override=True`.
    """
    env_path = Path(path).resolve() if path is not None else find_env_file()
    if env_path is None or not env_path.is_file():
        return {}

    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    parsed = _parse_env_text(text)
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    os.environ.setdefault("TPA_ENV_FILE", str(env_path))
    return applied


def env_status(path: str | Path | None = None) -> dict[str, object]:
    """Describe the effective environment without ever leaking secret values."""
    env_path = Path(path).resolve() if path is not None else find_env_file()
    keys = [
        "TPA_VISION_PROVIDER",
        "TPA_MODEL_NAME",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ]
    state: dict[str, object] = {
        "env_file": str(env_path) if env_path and env_path.is_file() else None,
        "provider": os.environ.get("TPA_VISION_PROVIDER", "offline"),
        "model_name": os.environ.get("TPA_MODEL_NAME"),
    }
    for key in keys[2:]:
        state[key] = "set" if os.environ.get(key) else "unset"
    return state
