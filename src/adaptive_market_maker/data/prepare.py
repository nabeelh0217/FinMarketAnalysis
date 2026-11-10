"""Dataset preparation: raw files -> validated, normalized Parquet + manifest.

Synthetic sessions are first written to disk in LOBSTER's exact layout and then read back through the
LOBSTER adapter, so the synthetic path exercises the same parser the real data will use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from adaptive_market_maker.config import DataConfig, SessionSpec
from adaptive_market_maker.data.base import Dataset, SessionData
from adaptive_market_maker.data.lobster import load_session, write_session
from adaptive_market_maker.data.manifest import build_manifest, write_manifest
from adaptive_market_maker.data.validate import ValidationReport, validate_session

log = logging.getLogger(__name__)


def session_specs(cfg: DataConfig) -> list[SessionSpec]:
    if cfg.source == "synthetic":
        from adaptive_market_maker.data.synthetic import synthetic_session_specs

        return synthetic_session_specs(cfg)
    return list(cfg.sessions)


def normalized_path(cfg: DataConfig, session_id: str) -> Path:
    return cfg.dataset_dir / f"{session_id}.parquet"


def generate_synthetic_raw(cfg: DataConfig, force: bool = False) -> list[Path]:
    from adaptive_market_maker.data.synthetic import generate_session

    written = []
    for idx, spec in enumerate(session_specs(cfg)):
        msg = spec.message_path(cfg.raw_dir, cfg.levels)
        if msg.exists() and not force:
            log.info("synthetic raw exists, skipping: %s", msg.name)
            continue
        session = generate_session(cfg, spec, idx)
        written += list(write_session(session, cfg.raw_dir, spec))
        log.info("generated synthetic session %s: %d messages", spec.session_id, len(session))
    return written


def validate_raw(cfg: DataConfig) -> list[ValidationReport]:
    reports = []
    for spec in session_specs(cfg):
        session = load_session(spec, cfg, source=cfg.source)
        reports.append(validate_session(session, cfg.max_gap_warning_s))
    return reports


def prepare_dataset(cfg: DataConfig, force: bool = False) -> list[Path]:
    """Parse, validate (fail loudly on critical issues), normalize and manifest every session."""
    if cfg.source == "synthetic":
        generate_synthetic_raw(cfg, force=force)
    out = []
    for spec in session_specs(cfg):
        target = normalized_path(cfg, spec.session_id)
        if target.exists() and not force:
            out.append(target)
            continue
        session = load_session(spec, cfg, source=cfg.source)
        report = validate_session(session, cfg.max_gap_warning_s, raise_on_critical=True)
        session.save(target)
        manifest = build_manifest(
            session,
            cfg,
            [spec.message_path(cfg.raw_dir, cfg.levels), spec.orderbook_path(cfg.raw_dir, cfg.levels)],
            target,
            report,
        )
        write_manifest(manifest, cfg)
        (cfg.manifest_dir / cfg.dataset_id / f"{spec.session_id}.validation.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str)
        )
        log.info("prepared %s (%s)", spec.session_id, report.status)
        out.append(target)
    return out


def load_dataset(cfg: DataConfig, prepare_if_missing: bool = True) -> Dataset:
    sessions: list[SessionData] = []
    for spec in session_specs(cfg):
        path = normalized_path(cfg, spec.session_id)
        if not path.exists():
            if not prepare_if_missing:
                raise FileNotFoundError(f"{path} missing; run `amm data prepare` first")
            prepare_dataset(cfg)
        sessions.append(SessionData.load(path))
    return Dataset(cfg.dataset_id, sessions)
