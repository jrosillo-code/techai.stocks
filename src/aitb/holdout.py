"""Holdout-lock discipline (tamper-evident).

The final holdout window may be evaluated ONCE, after the selected strategy
specifications are frozen. Enforcement:

  * ``freeze_selection`` hashes the frozen specs into the lock BEFORE any
    holdout numbers are computed;
  * ``record_holdout_access`` appends to the lock's access log AND mirrors an
    append-only event into the mode's experiment registry
    (results/<mode>/experiments.jsonl). The registry is the durable witness:
    deleting or rewriting holdout_lock.json does NOT clean the record,
    because ``holdout_status`` cross-checks the registry and marks the state
    COMPROMISED whenever the lock file shows fewer events than the registry
    (tamper-evidence by redundancy);
  * each event is hash-chained to the previous one and binds the study freeze
    hash and data-store fingerprint where available;
  * a second access, or access before freezing, flips ``compromised``
    permanently.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import results_dir
from .utils import stable_hash

LOCK_FILENAME = "holdout_lock.json"


def _path(mode: str) -> Path:
    d = results_dir(mode)
    d.mkdir(parents=True, exist_ok=True)
    return d / LOCK_FILENAME


def _registry_path(mode: str) -> Path:
    return results_dir(mode) / "experiments.jsonl"


def _registry_events(mode: str) -> list[dict]:
    p = _registry_path(mode)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "holdout_event":
            out.append(rec)
    return out


def _mirror_to_registry(mode: str, event: dict) -> None:
    p = _registry_path(mode)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps({"id": f"holdout_{event['chain']}",
                             "status": "holdout_event", "data_mode": mode,
                             **event}, default=str) + "\n")


def _load(mode: str) -> dict:
    p = _path(mode)
    return json.loads(p.read_text()) if p.exists() else {
        "frozen_specs_hash": None, "frozen_at": None, "specs": None,
        "access_log": [], "compromised": False, "violations": []}


def _save(mode: str, state: dict) -> None:
    _path(mode).write_text(json.dumps(state, indent=2, default=str))


def _chain(state: dict, payload: dict) -> str:
    prev = (state["access_log"][-1].get("chain") if state["access_log"] else
            state.get("frozen_specs_hash") or "genesis")
    return stable_hash([prev, payload], 16)


def _study_bindings() -> dict:
    """Freeze hash + data-store fingerprint, best-effort (never raises)."""
    out: dict = {}
    try:
        from .freeze import load_freeze
        out["freeze_hash"] = load_freeze()["hash"]
    except Exception:
        out["freeze_hash"] = None
    try:
        from .data.quality import store_fingerprint
        out["store_fingerprint"] = store_fingerprint()
    except Exception:
        out["store_fingerprint"] = None
    return out


def freeze_selection(specs: list[dict], mode: str, holdout_start: str) -> str:
    """Freeze the chosen strategy specs before looking at the holdout."""
    state = _load(mode)
    new_hash = stable_hash(specs, 16)
    if state["access_log"] and state["frozen_specs_hash"] not in (None, new_hash):
        state["violations"].append({
            "at": datetime.now(timezone.utc).isoformat(),
            "kind": "respecified_after_holdout_access",
            "detail": "selection changed after holdout was already viewed"})
        state["compromised"] = True
    state.update({"frozen_specs_hash": new_hash, "specs": specs,
                  "holdout_start": holdout_start,
                  "frozen_at": datetime.now(timezone.utc).isoformat()})
    event = {"kind": "freeze_selection", "specs_hash": new_hash,
             "at": state["frozen_at"], **_study_bindings()}
    event["chain"] = _chain(state, event)
    _mirror_to_registry(mode, event)
    _save(mode, state)
    return new_hash


def record_holdout_access(mode: str, purpose: str) -> dict:
    """Log a holdout evaluation (lock file + registry mirror). Returns state;
    caller must surface `compromised` in every report."""
    state = _load(mode)
    if state["frozen_specs_hash"] is None:
        state["violations"].append({
            "at": datetime.now(timezone.utc).isoformat(),
            "kind": "access_before_freeze",
            "detail": purpose})
        state["compromised"] = True
    event = {"kind": "access", "purpose": purpose,
             "at": datetime.now(timezone.utc).isoformat(), **_study_bindings()}
    event["chain"] = _chain(state, event)
    state["access_log"].append(event)
    if len(state["access_log"]) > 1:
        state["compromised"] = True
    _mirror_to_registry(mode, event)
    _save(mode, state)
    return state


def holdout_status(mode: str) -> dict:
    """Lock state cross-checked against the registry mirror.

    Fewer events in the lock than in the registry (file deleted/rewritten),
    or a chain value present in the registry but absent from the lock, means
    tampering: status is forced to compromised. The registry itself is
    append-only; truncating it also truncates experiment records, which the
    report provenance section would expose.
    """
    state = _load(mode)
    reg = _registry_events(mode)
    reg_accesses = [e for e in reg if e.get("kind") == "access"]
    lock_chains = {a.get("chain") for a in state["access_log"]}
    tampered = (len(reg_accesses) > len(state["access_log"]) or
                any(e.get("chain") not in lock_chains for e in reg_accesses))
    if tampered:
        state["compromised"] = True
        state.setdefault("violations", []).append({
            "kind": "lock_registry_mismatch",
            "detail": f"registry shows {len(reg_accesses)} access events, "
                      f"lock shows {len(state['access_log'])} — lock file was "
                      "deleted or rewritten"})
    if len(reg_accesses) > 1:
        state["compromised"] = True
    return state
