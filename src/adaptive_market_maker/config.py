"""Validated, file-based configuration.

Every material assumption (latency, fees, fill model, risk limits, horizons, splits, ...) lives here so
that it can be saved with each experiment and recovered later. Nothing that changes results may be
hard-coded inside strategy or execution logic.

Loading rules (see :func:`load_config`):

1. Section defaults are read from ``configs/<section>.yaml`` (``data``, ``features``, ``models``,
   ``execution``, ``strategies``).
2. If the requested file has an ``include`` list, each included file's top-level sections *replace* the
   corresponding sections wholesale (e.g. swap the LOBSTER data section for the synthetic one).
3. The requested file's own sections are then *deep-merged* on top (dicts merge, lists/scalars replace).
4. The result is validated; unknown keys are errors.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SECTION_FILES = ("data", "features", "models", "execution", "strategies")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def parse_clock(value: str) -> int:
    """'HH:MM:SS[.fff]' -> nanoseconds after midnight."""
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected HH:MM:SS, got {value!r}")
    h, m = int(parts[0]), int(parts[1])
    sec_str = parts[2]
    if "." in sec_str:
        whole, frac = sec_str.split(".")
        frac = (frac + "000000000")[:9]
    else:
        whole, frac = sec_str, "0"
    s = int(whole)
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 61):
        raise ValueError(f"invalid clock time {value!r}")
    return ((h * 60 + m) * 60 + s) * 1_000_000_000 + int(frac)


# ----------------------------------------------------------------------------------------------- data
class SessionSpec(StrictModel):
    """One trading session (one ticker, one date) of LOBSTER-format files."""

    ticker: str
    date: str
    start_ms: int = 34_200_000
    end_ms: int = 57_600_000
    message_file: str | None = None
    orderbook_file: str | None = None

    @property
    def session_id(self) -> str:
        return f"{self.ticker}_{self.date}"

    def default_stem(self, levels: int) -> str:
        return f"{self.ticker}_{self.date}_{self.start_ms}_{self.end_ms}"

    def message_path(self, raw_dir: Path, levels: int) -> Path:
        name = self.message_file or f"{self.default_stem(levels)}_message_{levels}.csv"
        return raw_dir / name

    def orderbook_path(self, raw_dir: Path, levels: int) -> Path:
        name = self.orderbook_file or f"{self.default_stem(levels)}_orderbook_{levels}.csv"
        return raw_dir / name


class SyntheticConfig(StrictModel):
    """Parameters of the Mode-2 synthetic order-level market (see data/synthetic.py).

    All rates are per second of market time. Sizes are in lots of ``lot_size`` shares. These values are
    illustrative, chosen to give a large-tick book (spread usually 1 tick); they are not calibrated to any
    real instrument.
    """

    scenario: Literal["informed", "null"] = "informed"
    ticker: str = "SYNTH"
    first_date: str = "2099-01-05"
    n_sessions: int = Field(6, ge=1)
    duration_s: float = Field(1800.0, gt=0)
    seed: int = 7
    initial_mid_ticks: int = Field(3000, gt=100)
    init_levels: int = 10
    init_orders_per_level: int = 3
    limit_rate: float = 20.0
    limit_distance_alpha: float = 0.8
    max_distance: int = 10
    market_rate: float = 1.5
    null_market_rate: float = Field(3.0, description="null scenario: uninformed flow replacing informed flow")
    cancel_rate_per_order: float = 0.3
    partial_cancel_prob: float = 0.2
    mean_limit_lots: float = 2.0
    mean_market_lots: float = 1.6
    fair_value_vol: float = Field(0.30, ge=0, description="latent value diffusion, ticks/sqrt(s)")
    jump_rate: float = 0.003
    jump_size_ticks: float = 2.0
    informed_rate: float = 3.0
    informed_scale_ticks: float = 0.5
    limit_tilt: float = 0.25

    @model_validator(mode="after")
    def _null_has_no_information(self) -> SyntheticConfig:
        if self.scenario == "null":
            # same total aggressive activity, none of it informed
            self.informed_rate = 0.0
            self.limit_tilt = 0.0
            self.market_rate = self.null_market_rate
        return self


class DataConfig(StrictModel):
    source: Literal["lobster", "synthetic"]
    dataset_id: str
    raw_dir: Path
    processed_dir: Path = Path("data/processed")
    manifest_dir: Path = Path("data/manifests")
    levels: int = Field(10, ge=1, le=50)
    price_scale: int = Field(10_000, description="integer price units per dollar (LOBSTER: $ x 10000)")
    tick_size: int = Field(100, gt=0, description="tick size in integer price units ($0.01 = 100)")
    lot_size: int = Field(100, gt=0, description="shares per round lot")
    open_time: str = "09:30:00"
    close_time: str = "16:00:00"
    sessions: list[SessionSpec] = Field(default_factory=list)
    synthetic: SyntheticConfig | None = None
    max_gap_warning_s: float = 30.0
    license_note: str = ""

    @model_validator(mode="after")
    def _check_source(self) -> DataConfig:
        if self.source == "synthetic":
            if self.synthetic is None:
                self.synthetic = SyntheticConfig()
            # a synthetic session closes when its simulated window ends
            close = self.open_ns + int(round(self.synthetic.duration_s * 1e9))
            secs, ns = divmod(close, 1_000_000_000)
            self.close_time = f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}.{ns:09d}"
        if self.source == "lobster" and not self.sessions:
            raise ValueError("data.sessions must list at least one LOBSTER session")
        if self.price_scale % 100 != 0:
            raise ValueError("price_scale must be a multiple of 100 (cash is kept in integer micro-dollars)")
        return self

    @property
    def open_ns(self) -> int:
        return parse_clock(self.open_time)

    @property
    def close_ns(self) -> int:
        return parse_clock(self.close_time)

    @property
    def dataset_dir(self) -> Path:
        return self.processed_dir / self.dataset_id


# ------------------------------------------------------------------------------------------- features
class FeatureConfig(StrictModel):
    depth_levels: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    depth_weights: Literal["uniform", "linear_decay", "exp_decay"] = "uniform"
    event_windows: list[int] = Field(default_factory=lambda: [10, 50, 200])
    time_windows_s: list[float] = Field(default_factory=lambda: [1.0, 5.0, 30.0])
    vol_window_s: float = 30.0
    vol_floor_ticks_per_sqrt_s: float = 0.01
    warmup_s: float = 60.0

    @field_validator("depth_levels", "event_windows")
    @classmethod
    def _positive_sorted(cls, v: list[int]) -> list[int]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("must be a non-empty list of positive integers")
        return sorted(set(v))


# ---------------------------------------------------------------------------------------------- models
class LabelConfig(StrictModel):
    kind: Literal["event", "time"] = "time"
    horizon_events: int = Field(10, ge=1, description="horizon in observable book states (decision rows)")
    horizon_ms: float = Field(1000.0, gt=0)
    classes: Literal["ternary", "binary_up"] = "ternary"
    flat_threshold_ticks: float = Field(0.0, ge=0)


class SplitConfig(StrictModel):
    mode: Literal["intraday", "sessions"] = "intraday"
    train_frac: float = Field(0.5, gt=0, lt=1)
    val_frac: float = Field(0.2, gt=0, lt=1)
    train_sessions: int = 3
    val_sessions: int = 1
    embargo_s: float = Field(30.0, ge=0)
    exclude_open_s: float = Field(300.0, ge=0)
    exclude_close_s: float = Field(300.0, ge=0)

    @model_validator(mode="after")
    def _fractions(self) -> SplitConfig:
        if self.mode == "intraday" and self.train_frac + self.val_frac >= 0.95:
            raise ValueError("train_frac + val_frac must leave room for a test period")
        return self


class FillModelConfig(StrictModel):
    horizon_s: float = Field(10.0, gt=0)
    n_shadow_orders: int = Field(4000, ge=100)
    seed: int = 11


class ModelConfig(StrictModel):
    labels: LabelConfig = Field(default_factory=LabelConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    fill_model: FillModelConfig = Field(default_factory=FillModelConfig)
    candidates: list[Literal["prior", "logistic", "hgb"]] = Field(
        default_factory=lambda: ["prior", "logistic", "hgb"]
    )
    feature_groups: list[str] | None = None
    logistic_C: float = Field(0.5, gt=0)
    logistic_max_iter: int = 1000
    hgb_max_iter: int = 150
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 15
    hgb_min_samples_leaf: int = 200
    hgb_l2: float = 1.0
    calibrate: bool = True
    eval_stride: Literal["non_overlapping", "all"] = "non_overlapping"
    max_train_samples: int = 200_000
    seed: int = 0


# ------------------------------------------------------------------------------------------- execution
class LatencyConfig(StrictModel):
    """Constant one-way latencies in microseconds. Illustrative, not measured."""

    market_data_us: float = Field(250.0, ge=0, description="exchange event -> strategy observes it")
    decision_us: float = Field(20.0, ge=0, description="strategy compute time")
    order_entry_us: float = Field(250.0, ge=0, description="strategy -> order/cancel reaches venue")
    report_us: float = Field(250.0, ge=0, description="venue fill/ack -> strategy learns of it")


class FeeConfig(StrictModel):
    """Per-share fees in dollars. Negative = rebate. Defaults are ILLUSTRATIVE, not a verified schedule."""

    maker_per_share: float = -0.0020
    taker_per_share: float = 0.0030


class ExecutionConfig(StrictModel):
    fill_preset: Literal["conservative", "central", "optimistic"] = "central"
    conservative_extra_queue_frac: float = Field(
        0.0, ge=0, description="conservative preset: extra queue ahead as a fraction of displayed size"
    )
    post_only: bool = True
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    fees: FeeConfig = Field(default_factory=FeeConfig)


class RiskConfig(StrictModel):
    max_inventory: int = Field(500, gt=0, description="absolute position limit, shares")
    max_order_size: int = Field(200, gt=0)
    max_outstanding_per_side: int = Field(200, gt=0)
    max_notional: float = Field(250_000.0, gt=0, description="dollars")
    max_actions_per_second: float = Field(20.0, gt=0)
    max_quote_age_s: float | None = None
    max_market_spread_ticks: int = Field(10, gt=0)
    max_price_deviation_ticks: int = Field(10, gt=0)
    max_session_loss: float = Field(2_000.0, gt=0, description="dollars, mark-to-mid; kill switch")
    stale_data_s: float = Field(5.0, gt=0)


# ------------------------------------------------------------------------------------------ strategies
class QuoteSizing(StrictModel):
    quote_size: int = Field(100, gt=0)
    requote_threshold_ticks: int = Field(0, ge=0, description="keep resting order if within this many ticks")
    rounding: Literal["nearest", "conservative"] = "nearest"


class FixedSpreadParams(QuoteSizing):
    half_spread_ticks: float = Field(0.5, ge=0)


class InventoryAwareParams(QuoteSizing):
    gamma: float = Field(0.1, gt=0, description="risk aversion, 1/(tick*lot)")
    k: float = Field(1.5, gt=0, description="fill-intensity decay, 1/tick")
    A: float = Field(1.0, gt=0, description="fill intensity at zero distance, fills/s (GLFT only)")
    horizon_mode: Literal["rolling", "session_end", "glft"] = "rolling"
    tau_s: float = Field(30.0, gt=0, description="rolling inventory-risk horizon, seconds")
    spread_mode: Literal["model", "fixed"] = "fixed"
    half_spread_ticks: float = Field(0.5, ge=0, description="used when spread_mode == fixed")
    min_half_spread_ticks: float = Field(0.5, ge=0)
    use_vol_adaptation: bool = True
    static_vol_ticks_per_sqrt_s: float = Field(0.1, gt=0, description="used when vol adaptation is off")
    use_inventory_skew: bool = True
    estimate_fill_intensity: bool = Field(
        False, description="replace k and A with estimates from training-period shadow orders"
    )


class AdaptiveParams(InventoryAwareParams):
    alpha_weight: float = Field(1.0, ge=0)
    alpha_clip_ticks: float = Field(2.0, gt=0)
    alpha_guard_ticks: float | None = Field(
        0.25, description="withdraw the side that alpha says will be adversely selected; None disables"
    )
    use_alpha: bool = True


class RegimeParams(AdaptiveParams):
    high_vol_ticks_per_sqrt_s: float = 0.5
    wide_spread_ticks: int = 3
    high_vol_size_mult: float = 0.0
    wide_spread_extra_ticks: float = 0.5


class StrategiesConfig(StrictModel):
    fixed_spread: FixedSpreadParams = Field(default_factory=FixedSpreadParams)
    inventory_aware: InventoryAwareParams = Field(default_factory=InventoryAwareParams)
    adaptive: AdaptiveParams = Field(default_factory=AdaptiveParams)
    regime: RegimeParams = Field(default_factory=RegimeParams)
    risk: RiskConfig = Field(default_factory=RiskConfig)


class BacktestConfig(StrictModel):
    flatten_before_end_s: float = Field(30.0, ge=0)
    equity_interval_s: float = Field(1.0, gt=0)
    decision_min_interval_ms: float = Field(0.0, ge=0)
    markout_horizons_ms: list[float] = Field(default_factory=lambda: [100.0, 1000.0, 5000.0, 30000.0])
    bootstrap_block_s: float = Field(60.0, gt=0, description="clock interval for P&L aggregation")
    bootstrap_blocks_per_draw: int = Field(3, ge=1, description="consecutive intervals per moving block")
    n_bootstrap: int = Field(2000, ge=100)
    seed: int = 0


# ------------------------------------------------------------------------------------------ experiment
class ExperimentConfig(StrictModel):
    id: str = "default"
    description: str = ""
    kind: Literal["exploratory", "confirmatory"] = "exploratory"
    strategies: list[Literal["fixed_spread", "inventory_aware", "adaptive", "regime"]] = Field(
        default_factory=lambda: ["fixed_spread", "inventory_aware", "adaptive"]
    )
    validation_grid: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)
    sensitivity: dict[str, list[Any]] = Field(default_factory=dict)
    sensitivity_strategies: list[str] = Field(
        default_factory=lambda: ["fixed_spread", "inventory_aware", "adaptive"]
    )
    ablations: list[str] = Field(default_factory=list)
    output_dir: Path = Path("reports/experiments")
    overwrite: bool = False
    n_jobs: int = Field(1, ge=1)


class ProjectConfig(StrictModel):
    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        blob = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------- loading
def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as fh:
        content = yaml.safe_load(fh) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path}: top level must be a mapping of section names")
    return content


def config_root() -> Path:
    return Path(os.environ.get("AMM_CONFIG_DIR", "configs"))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ProjectConfig:
    root = config_root()
    merged: dict[str, Any] = {}
    for section in SECTION_FILES:
        f = root / f"{section}.yaml"
        if f.exists():
            merged = deep_merge(merged, _read_yaml(f))
    if path is not None:
        main = _read_yaml(Path(path))
        for inc in main.pop("include", []) or []:
            for section, body in _read_yaml(Path(inc)).items():
                merged[section] = copy.deepcopy(body)
        merged = deep_merge(merged, main)
    if overrides:
        merged = deep_merge(merged, overrides)
    if "data" not in merged:
        raise ValueError(f"no data section found (looked in {root}/data.yaml and {path})")
    return ProjectConfig.model_validate(merged)


def set_by_path(tree: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``tree`` with ``a.b.c`` set to ``value`` (used by sensitivity/ablation grids)."""
    out = copy.deepcopy(tree)
    node = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return out


def with_overrides(cfg: ProjectConfig, assignments: dict[str, Any]) -> ProjectConfig:
    tree = cfg.snapshot()
    for dotted, value in assignments.items():
        tree = set_by_path(tree, dotted, value)
    return ProjectConfig.model_validate(tree)


def save_snapshot(cfg: ProjectConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(cfg.snapshot(), fh, sort_keys=False)
