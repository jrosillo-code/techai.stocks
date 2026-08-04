"""Plugin architecture for future strategy families.

A plugin is a directory under ``plugins/<name>/`` containing ``strategy.py``
that defines one or more ``aitb.strategies.base.Strategy`` subclasses and
calls ``register()`` on each. Registered classes appear automatically in the
catalog, documentation, dashboard, comparison engine and (if portable) the
TradingView export — no other wiring.

GOVERNANCE: plugins are NOT part of the research freeze. They may run freely
in synthetic mode; using one in a REAL study requires (a) adding its module
to the freeze's fingerprint list and its grid to experiments.yaml, and (b) a
freeze version bump. ``load_plugins`` enforces the soft side of this: plugin
classes are tagged ``is_plugin`` and the real-mode runner only executes grids
from the frozen experiments.yaml — an unfrozen plugin can never slip into a
frozen study because its grid is not in the frozen config (and adding it
changes the freeze hash).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .config import PROJECT_ROOT
from .strategies import STRATEGY_CLASSES, Strategy
from .utils import get_logger

log = get_logger("plugins")

PLUGIN_DIR = PROJECT_ROOT / "plugins"


def register(cls: type) -> type:
    """Decorator/function: add a Strategy subclass to the global registry."""
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise TypeError(f"{cls} is not a Strategy subclass")
    cls.is_plugin = True
    STRATEGY_CLASSES[cls.__name__] = cls
    log.info("registered plugin strategy %s", cls.__name__)
    return cls


def load_plugins() -> list[str]:
    """Import every plugins/<name>/strategy.py. Returns loaded plugin names."""
    loaded = []
    if not PLUGIN_DIR.exists():
        return loaded
    for d in sorted(PLUGIN_DIR.iterdir()):
        mod_path = d / "strategy.py"
        if not d.is_dir() or not mod_path.exists():
            continue
        name = f"aitb_plugin_{d.name}"
        try:
            spec = importlib.util.spec_from_file_location(name, mod_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded.append(d.name)
        except Exception as exc:
            log.warning("plugin %s failed to load: %s", d.name, exc)
    return loaded
