"""Load API keys from a .env file at the repo root.

Keys live in .env (git-ignored) rather than in the shell, so every teammate
gets the same setup with no per-machine profile editing, and no key ever
reaches a commit.

A real environment variable always wins over the file, so CI or a one-off
`GEMINI_API_KEY=... python ...` can override without editing anything.
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def find_dotenv(start: Path | None = None) -> Path | None:
    """Walk up from here looking for a .env file."""
    here = (start or Path(__file__).resolve()).parent
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Minimal KEY=VALUE parsing -- comments, blanks and quotes only."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Read .env into ``os.environ``. Safe to call repeatedly."""
    global _loaded
    path = path or find_dotenv()
    if path is None or not path.is_file():
        return {}
    values = parse_dotenv(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    _loaded = True
    return values
