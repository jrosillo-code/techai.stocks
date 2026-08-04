"""Report generation: charts (PNG/SVG) and a self-contained HTML report."""
from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from .config import REPORTS_DIR
from .metrics import drawdown_series, rolling_sharpe
from .utils import get_logger

log = get_logger("reporting")

plt.rcParams.update({
    "figure.figsize": (11, 5), "figure.dpi": 110, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def equity_chart(curves: dict[str, pd.Series], title: str, logy: bool = True) -> str:
    fig, ax = plt.subplots()
    for name, eq in curves.items():
        norm = eq / eq.dropna().iloc[0]
        ax.plot(norm.index, norm.values, label=name, linewidth=1.2)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=2)
    ax.set_ylabel("Growth of $1 (log scale)" if logy else "Growth of $1")
    return _fig_to_b64(fig)


def drawdown_chart(returns: dict[str, pd.Series], title: str) -> str:
    fig, ax = plt.subplots()
    for name, r in returns.items():
        dd = drawdown_series(r)
        ax.plot(dd.index, dd.values, label=name, linewidth=1.0)
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=2)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return _fig_to_b64(fig)


def rolling_sharpe_chart(returns: dict[str, pd.Series], title: str,
                         window_years: int = 3) -> str:
    fig, ax = plt.subplots()
    for name, r in returns.items():
        rs = rolling_sharpe(r, window_years)
        ax.plot(rs.index, rs.values, label=name, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{title} — rolling {window_years}y Sharpe")
    ax.legend(fontsize=8, ncols=2)
    return _fig_to_b64(fig)


def annual_returns_heatmap(returns: dict[str, pd.Series], title: str) -> str:
    tbl = pd.DataFrame({
        name: (1 + r).resample("YE").prod() - 1 for name, r in returns.items()
    })
    tbl.index = tbl.index.year
    fig, ax = plt.subplots(figsize=(11, max(3, 0.35 * len(tbl))))
    data = tbl.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(tbl.columns)), tbl.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tbl.index)), tbl.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, shrink=0.6, format=lambda v, _: f"{v:.0%}")
    return _fig_to_b64(fig)


def sensitivity_heatmap(df: pd.DataFrame, x: str, y: str, z: str, title: str) -> str:
    pivot = df.pivot_table(index=y, columns=x, values=z)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, shrink=0.7)
    return _fig_to_b64(fig)


def correlation_heatmap(corr: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    labels = [c[:28] for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, shrink=0.7)
    return _fig_to_b64(fig)


_TEMPLATE = Template("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
 body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:2rem auto;max-width:1100px;
      color:#1a1a1a;line-height:1.45;padding:0 1rem}
 h1{border-bottom:3px solid #2563eb;padding-bottom:.3rem}
 h2{margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
 table{border-collapse:collapse;font-size:.82rem;margin:.8rem 0;width:100%}
 th,td{border:1px solid #ddd;padding:.28rem .5rem;text-align:right}
 th{background:#f1f5f9}
 td:first-child,th:first-child{text-align:left}
 img{max-width:100%;border:1px solid #eee;margin:.5rem 0}
 .warn{background:#fef3c7;border-left:4px solid #f59e0b;padding:.7rem 1rem;margin:1rem 0}
 .note{color:#555;font-size:.85rem}
 .robust{color:#166534;font-weight:600}.inconclusive{color:#92400e}.rejected{color:#991b1b}
</style></head><body>
<h1>{{ title }}</h1>
<p class="note">Generated {{ generated }} · data provider: <b>{{ provider }}</b> ·
data span {{ span }} · git {{ git }}</p>
{% for w in warnings %}<div class="warn">{{ w }}</div>{% endfor %}
{% for section in sections %}
<h2>{{ section.title }}</h2>
{{ section.html }}
{% endfor %}
</body></html>""")


def render_report(title: str, sections: list[dict], provider: str, span: str,
                  git: str, warnings: list[str],
                  out_path: Path | None = None) -> Path:
    html = _TEMPLATE.render(
        title=title, sections=sections, provider=provider, span=span, git=git,
        warnings=warnings,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    out = out_path or REPORTS_DIR / "research_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    log.info("report written to %s", out)
    return out


def df_to_html(df: pd.DataFrame, pct_cols: tuple = (), float_fmt: str = "{:.2f}") -> str:
    d = df.copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
    return d.to_html(index=False, escape=True, border=0)


def img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}">'
