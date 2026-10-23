"""End-to-end: synthetic events -> features -> models -> strategies -> fills -> accounting -> report, and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_market_maker.cli import main
from adaptive_market_maker.config import load_config, with_overrides
from adaptive_market_maker.experiments.runner import run_experiment
from adaptive_market_maker.reporting.report import build_report

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def smoke_cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = load_config(REPO / "configs/experiments/smoke.yaml")
    return with_overrides(
        cfg,
        {
            "data.raw_dir": str(tmp_path / "raw"),
            "data.processed_dir": str(tmp_path / "processed"),
            "data.manifest_dir": str(tmp_path / "manifests"),
            "experiment.output_dir": str(tmp_path / "experiments"),
        },
    )


@pytest.mark.slow
def test_full_protocol_runs_and_report_is_generated(smoke_cfg):
    out = run_experiment(smoke_cfg)
    res = json.loads((out / "results.json").read_text())
    assert res["synthetic"] is True
    assert set(res["test"]["by_strategy"]) == {"fixed_spread", "inventory_aware", "adaptive"}
    # every stage used consistent units: P&L decomposes into gross minus fees
    for d in res["test"]["by_strategy"].values():
        assert d["gross_pnl_total"] - d["fees_net_total"] == pytest.approx(d["net_pnl_total"], abs=1e-6)
    # C without alpha is B exactly (identical inherited parameters)
    assert res["ablations"]["trading"]["no_alpha"]["net_pnl"] == pytest.approx(
        res["test"]["by_strategy"]["inventory_aware"]["net_pnl_total"]
    )
    assert (out / "trials.csv").exists() and (out / "config_selected.yaml").exists()
    assert (out.parent / "registry.jsonl").exists()
    report = build_report(out)
    text = report.read_text()
    assert "SYNTHETIC DATA" in text and "## 19. Reproduction" in text
    assert len(list((out / "figures").glob("*.png"))) >= 12
    with pytest.raises(FileExistsError):  # never overwrite results silently
        run_experiment(smoke_cfg)


def test_cli_help_and_data_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(REPO)
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    cfg = tmp_path / "data.yaml"
    cfg.write_text(
        "data:\n  source: synthetic\n  dataset_id: cli_test\n"
        f"  raw_dir: {tmp_path / 'raw'}\n  processed_dir: {tmp_path / 'p'}\n  manifest_dir: {tmp_path / 'm'}\n"
        "  sessions: []\n  synthetic: {n_sessions: 1, duration_s: 120}\n"
    )
    assert main(["data", "validate", "--config", str(cfg)]) == 0
    assert "book_transitions_consistent" in capsys.readouterr().out
    assert main(["data", "prepare", "--config", str(cfg)]) == 0
    assert list((tmp_path / "p" / "cli_test").glob("*.parquet"))


def test_cli_reports_missing_lobster_files_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(REPO)
    cfg = tmp_path / "d.yaml"
    cfg.write_text(
        f"data:\n  source: lobster\n  dataset_id: x\n  raw_dir: {tmp_path}\n"
        "  sessions: [{ticker: MSFT, date: '2012-06-21'}]\n"
    )
    assert main(["data", "validate", "--config", str(cfg)]) == 2
    assert "LOBSTER" in capsys.readouterr().err


LOBSTER = REPO / "data/raw/lobster/MSFT_2012-06-21_34200000_57600000_message_10.csv"


@pytest.mark.needs_lobster
@pytest.mark.skipif(
    not LOBSTER.exists(), reason="real LOBSTER sample not present in data/raw/lobster (see README)"
)
def test_real_lobster_sample_validates(monkeypatch):
    monkeypatch.chdir(REPO)
    from adaptive_market_maker.data.lobster import load_session
    from adaptive_market_maker.data.validate import validate_session

    cfg = load_config(REPO / "configs/data.yaml")
    rep = validate_session(load_session(cfg.data.sessions[0], cfg.data))
    assert rep.passed, rep.format()


def test_parallel_jobs_fall_back_to_serial_without_fork(monkeypatch):
    """Windows has no fork start method; the runner must still work (serially)."""
    import multiprocessing as mp

    from adaptive_market_maker.experiments import runner

    calls = []
    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["spawn"])
    monkeypatch.setattr(runner, "_run_job", lambda job: calls.append(job) or {"ok": job})
    out = runner.run_jobs(ctx=object(), jobs=[1, 2, 3], n_jobs=4)
    assert out == [{"ok": 1}, {"ok": 2}, {"ok": 3}] and calls == [1, 2, 3]
