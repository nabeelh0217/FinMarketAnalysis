"""Dataset manifests: provenance, checksums, coverage and validation status for each session."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from adaptive_market_maker.config import DataConfig
from adaptive_market_maker.data.base import SCHEMA_VERSION, SessionData
from adaptive_market_maker.data.validate import ValidationReport
from adaptive_market_maker.types import NS_PER_S


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(
    session: SessionData, cfg: DataConfig, raw_files: list[Path], normalized: Path, report: ValidationReport
) -> dict:
    return {
        "dataset_id": cfg.dataset_id,
        "session_id": session.session_id,
        "source": session.source,
        "synthetic": session.source == "synthetic",
        "instrument": session.ticker,
        "trading_date": session.date,
        "levels": session.levels,
        "acquired_or_generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "raw_files": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in raw_files
        ],
        "normalized_file": {"path": str(normalized), "sha256": sha256_file(normalized)},
        "schema_version": SCHEMA_VERSION,
        "row_count": len(session),
        "timestamp_range_s": [float(session.ts[0] / NS_PER_S), float(session.ts[-1] / NS_PER_S)]
        if len(session)
        else None,
        "timestamp_semantics": "exchange event time, seconds after midnight (LOBSTER), parsed exactly to ns",
        "validation_status": report.status,
        "validation_failed_checks": [c.name for c in report.checks if not c.passed],
        "license": cfg.license_note,
        "known_limitations": [
            "messages beyond the recorded depth are absent from the file",
            "hidden/reserve liquidity is not visible until executed",
        ]
        + (["SYNTHETIC DATA: not evidence about real markets"] if session.source == "synthetic" else []),
        "normalization": {
            k: v for k, v in session.meta.items() if k not in ("message_file", "orderbook_file")
        },
    }


def write_manifest(manifest: dict, cfg: DataConfig) -> Path:
    out = cfg.manifest_dir / cfg.dataset_id / f"{manifest['session_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return out


def read_manifests(cfg: DataConfig) -> list[dict]:
    d = cfg.manifest_dir / cfg.dataset_id
    if not d.exists():
        return []
    return [
        json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if not p.name.endswith(".validation.json")
    ]
