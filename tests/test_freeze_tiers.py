"""Freeze integrity, tier assignment, and decision-brief conservatism."""
import json
from pathlib import Path

import pytest

from aitb import freeze as fz
from aitb.tiers import assign_tier


def test_canonical_build_is_deterministic():
    a = fz.freeze_hash(fz.build_canonical())
    b = fz.freeze_hash(fz.build_canonical())
    assert a == b and len(a) == 32


def test_freeze_create_verify_and_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(fz, "FREEZE_PATH", tmp_path / "research_freeze_v1.json")
    fz.create_freeze()
    doc = fz.verify_freeze()          # matches immediately after creation
    assert doc["hash"] == fz.freeze_hash(fz.build_canonical())
    # A freeze is immutable: creating again must refuse.
    with pytest.raises(FileExistsError):
        fz.create_freeze()
    # Tamper with the frozen hash -> verification must fail loudly.
    tampered = json.loads(fz.FREEZE_PATH.read_text())
    tampered["hash"] = "0" * 32
    fz.FREEZE_PATH.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="FREEZE VIOLATION"):
        fz.verify_freeze()


def test_freeze_detects_config_drift(tmp_path, monkeypatch):
    """Changing a strategy grid after freezing must abort real runs."""
    monkeypatch.setattr(fz, "FREEZE_PATH", tmp_path / "research_freeze_v1.json")
    fz.create_freeze()
    real_build = fz.build_canonical

    def drifted():
        c = real_build()
        c["experiments"]["tsmom"][0]["grid"]["sma_window"] = [199]  # "tuned"
        return c
    monkeypatch.setattr(fz, "build_canonical", drifted)
    with pytest.raises(RuntimeError, match="experiments"):
        fz.verify_freeze()


def test_missing_freeze_blocks():
    orig = fz.FREEZE_PATH
    try:
        fz.FREEZE_PATH = Path("/nonexistent/freeze.json")
        with pytest.raises(FileNotFoundError, match="no research freeze"):
            fz.verify_freeze()
    finally:
        fz.FREEZE_PATH = orig


# ------------------------------------------------------------------ tiers ---
GATE_LIMITED = {"limitations": [
    "only 0/5 configured delisted names have real history — the universe is "
    "PARTIALLY current-constituent biased (delisted names)",
    "fundamentals cover only 3 names",
    "no VIX series — VIX-based regime strategies must be skipped",
]}
GATE_CLEAN = {"limitations": []}


def test_tier_assignment():
    t, _ = assign_tier({"family": "riskmanaged", "status": "ok"}, GATE_CLEAN)
    assert t == "A"
    t, r = assign_tier({"family": "xsmom", "status": "ok"}, GATE_LIMITED)
    assert t == "B" and "delisted" in r
    t, _ = assign_tier({"family": "regime", "status": "ok"}, GATE_LIMITED)
    assert t == "C"          # VIX absent -> unavailable
    t, _ = assign_tier({"family": "regime", "status": "ok"}, GATE_CLEAN)
    assert t == "B"          # macro is always revised -> never A
    t, _ = assign_tier({"family": "fundamental", "status": "ok"}, GATE_LIMITED)
    assert t == "B"
    t, r = assign_tier({"family": "xsmom", "status": "deprecated"}, GATE_CLEAN)
    assert t == "C" and "deprecated" in r


def test_decision_brief_cannot_recommend_trading():
    """The brief's decision space is paper-trade or do-nothing, never trade."""
    src = Path("scripts/make_decision_brief.py").read_text()
    assert 'decision = "PAPER-TRADE' in src
    assert '"TRADE"' not in src.replace("PAPER-TRADE", "")
    assert "Live trading is out of scope" in src
