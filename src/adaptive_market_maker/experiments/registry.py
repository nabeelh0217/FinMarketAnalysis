"""Append-only experiment registry (JSON lines) so every evaluated configuration stays on record."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def append_entry(registry: Path, entry: dict) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": git_commit(),
        **entry,
    }
    with registry.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_entries(registry: Path) -> list[dict]:
    if not registry.exists():
        return []
    return [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
