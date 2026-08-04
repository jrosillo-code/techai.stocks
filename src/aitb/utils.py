"""Shared utilities: logging, hashing, deterministic seeding."""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from typing import Any

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("aitb")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(f"aitb.{name}")


def stable_hash(obj: Any, length: int = 10) -> str:
    """Deterministic short hash of any JSON-serializable object."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]
