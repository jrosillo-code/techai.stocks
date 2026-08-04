import numpy as np
import pandas as pd

from aitb import metrics as m


def _series(vals):
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)))


def test_total_return_and_cagr():
    r = _series([0.01] * 252)
    assert abs(m.total_return(r) - (1.01 ** 252 - 1)) < 1e-9
    # CAGR is annualized over ELAPSED CALENDAR TIME (audit AUD-006), not a
    # fixed 252-rows-per-year convention.
    years = (r.index[-1] - r.index[0]).days / 365.25
    expected = (1.01 ** 252) ** (1 / years) - 1
    assert abs(m.cagr(r) - expected) < 1e-9


def test_sharpe_of_constant_positive_is_undefined_or_large():
    r = _series(np.full(252, 0.001) + np.array([1e-6 * ((-1) ** i) for i in range(252)]))
    assert m.sharpe(r) > 10


def test_max_drawdown_known_path():
    # +10%, then -50%, then flat: max drawdown is exactly -50% from the peak.
    r = _series([0.10, -0.50] + [0.0] * 58)
    assert abs(m.max_drawdown(r) - (-0.50)) < 1e-9


def test_var_cvar_ordering():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0, 0.01, 1000))
    var, cvar = m.var_cvar(r)
    assert cvar <= var < 0


def test_beta_alpha_of_identical_series():
    rng = np.random.default_rng(1)
    r = _series(rng.normal(0.0005, 0.01, 500))
    beta, alpha = m.beta_alpha(r, r)
    assert abs(beta - 1) < 1e-9
    assert abs(alpha) < 1e-9


def test_capture_symmetric():
    rng = np.random.default_rng(2)
    b = _series(rng.normal(0, 0.01, 500))
    up, dn = m.capture(b, b)
    assert abs(up - 1) < 1e-9 and abs(dn - 1) < 1e-9


def test_summary_has_all_keys():
    rng = np.random.default_rng(3)
    r = _series(rng.normal(0.0005, 0.01, 600))
    s = m.summary(r, bench=r * 0.9, turnover_ann=2.0)
    for k in ("cagr", "sharpe", "sortino", "calmar", "max_drawdown", "var_95",
              "cvar_95", "beta", "alpha_ann", "information_ratio",
              "upside_capture", "downside_capture", "annual_turnover",
              "best_year", "worst_year", "profit_factor", "win_rate"):
        assert k in s, k
