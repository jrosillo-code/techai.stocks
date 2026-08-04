"""Configuration loading.

All tunable parameters live in YAML files under ``configs/``; code never
hard-codes universe membership, cost assumptions, or validation windows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("AITB_CONFIG_DIR", PROJECT_ROOT / "configs"))
DATA_DIR = Path(os.environ.get("AITB_DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("AITB_RESULTS_DIR", PROJECT_ROOT / "results"))
REPORTS_DIR = Path(os.environ.get("AITB_REPORTS_DIR", PROJECT_ROOT / "reports"))

# ---------------------------------------------------------------- data modes --
# 'real' and 'synthetic' runs are fully separated: distinct data roots,
# result registries and report directories. Nothing may cross between them —
# the loader enforces the data side, the registry/report paths the output side.
DATA_MODES = ("real", "synthetic")
REAL_DATA_DIR = DATA_DIR / "real"          # canonical validated real datasets
IMPORT_DIR = DATA_DIR / "import"           # user-supplied bundles land here


def results_dir(mode: str) -> Path:
    if mode not in DATA_MODES:
        raise ValueError(f"unknown data mode '{mode}'")
    return RESULTS_DIR / mode


def reports_dir(mode: str) -> Path:
    if mode not in DATA_MODES:
        raise ValueError(f"unknown data mode '{mode}'")
    return REPORTS_DIR / mode


def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name) as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Security:
    ticker: str
    name: str
    sector: str
    ipo: date
    delisted: date | None = None


@dataclass(frozen=True)
class Benchmark:
    ticker: str
    name: str
    ipo: date
    expense_ratio: float = 0.0


@dataclass
class UniverseConfig:
    securities: list[Security]
    benchmarks: list[Benchmark]
    baskets: dict[str, list[str]]
    min_price: float
    min_median_dollar_volume: float
    seasoning_days: int
    max_names: int

    @property
    def tickers(self) -> list[str]:
        return [s.ticker for s in self.securities]

    @property
    def benchmark_tickers(self) -> list[str]:
        return [b.ticker for b in self.benchmarks]

    def security(self, ticker: str) -> Security:
        for s in self.securities:
            if s.ticker == ticker:
                return s
        raise KeyError(ticker)


def _to_date(v: Any) -> date | None:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


def load_universe_config() -> UniverseConfig:
    raw = load_yaml("universe.yaml")
    securities = [
        Security(
            ticker=s["ticker"],
            name=s["name"],
            sector=s["sector"],
            ipo=_to_date(s["ipo"]),
            delisted=_to_date(s.get("delisted")),
        )
        for s in raw["securities"]
    ]
    benchmarks = [
        Benchmark(
            ticker=b["ticker"],
            name=b["name"],
            ipo=_to_date(b["ipo"]),
            expense_ratio=float(b.get("expense_ratio", 0.0)),
        )
        for b in raw["benchmarks"]
    ]
    st = raw["settings"]
    return UniverseConfig(
        securities=securities,
        benchmarks=benchmarks,
        baskets=raw.get("baskets", {}),
        min_price=float(st["min_price"]),
        min_median_dollar_volume=float(st["min_median_dollar_volume"]),
        seasoning_days=int(st["seasoning_days"]),
        max_names=int(st["max_names"]),
    )


@dataclass(frozen=True)
class CostScenario:
    name: str
    label: str
    commission_bps: float
    half_spread_bps: float
    slippage_bps: float
    impact_coeff_bps: float
    borrow_bps: float

    @property
    def fixed_one_way_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.slippage_bps


def load_cost_scenarios() -> dict[str, CostScenario]:
    raw = load_yaml("costs.yaml")["scenarios"]
    return {
        name: CostScenario(name=name, **{k: (v if k == "label" else float(v)) for k, v in cfg.items()})
        for name, cfg in raw.items()
    }


@dataclass
class BacktestConfig:
    initial_capital: float
    execution: str
    default_cost_scenario: str
    min_train_months: int
    step_months: int
    holdout_months: int
    n_bootstrap: int
    seed: int
    subperiods: dict[str, tuple[date, date]] = field(default_factory=dict)


def load_backtest_config() -> BacktestConfig:
    raw = load_yaml("backtest.yaml")
    eng, val = raw["engine"], raw["validation"]
    subs = {
        name: (_to_date(lo), _to_date(hi))
        for name, (lo, hi) in raw.get("subperiods", {}).items()
    }
    return BacktestConfig(
        initial_capital=float(eng["initial_capital"]),
        execution=eng["execution"],
        default_cost_scenario=eng["default_cost_scenario"],
        min_train_months=int(val["min_train_months"]),
        step_months=int(val["step_months"]),
        holdout_months=int(val["holdout_months"]),
        n_bootstrap=int(val["n_bootstrap"]),
        seed=int(val["seed"]),
        subperiods=subs,
    )
