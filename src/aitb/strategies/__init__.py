"""Strategy registry: every runnable strategy family lives here."""
from .base import Strategy
from . import (allocation, benchmarks, breakout, factor, fundamental,
               longshort, meanrev, ml, regime, riskmanaged, tsmom, xsmom)

__all__ = ["Strategy", "allocation", "benchmarks", "breakout", "factor",
           "fundamental", "longshort", "meanrev", "ml", "regime",
           "riskmanaged", "tsmom", "xsmom", "STRATEGY_CLASSES", "from_spec"]

STRATEGY_CLASSES: dict[str, type] = {}
for _mod in (benchmarks, tsmom, xsmom, meanrev, breakout, fundamental, regime,
             riskmanaged, ml, factor, allocation, longshort):
    for _name in dir(_mod):
        _obj = getattr(_mod, _name)
        if isinstance(_obj, type) and issubclass(_obj, Strategy) and _obj is not Strategy:
            STRATEGY_CLASSES[_name] = _obj


def from_spec(spec: dict) -> Strategy:
    """Rebuild a strategy instance from a registry record's spec."""
    cls = STRATEGY_CLASSES[spec["class"]]
    return cls(**spec.get("params", {}))
