# %% [markdown]
# # Interactive strategy comparison
# Open this file in Jupyter (via jupytext) or VS Code interactive mode.
# It loads the experiment registry and lets you compare any set of strategies
# against benchmarks, after the pipeline has run.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitb.experiments import ExperimentRegistry
from aitb.metrics import summary
from aitb.ranking import rank_experiments

registry = ExperimentRegistry()
df = registry.load()
ok = df[(df["status"] == "ok") & (df["scenario"] == "base")]
print(f"{len(ok)} experiments (base costs). Families: {sorted(ok['family'].unique())}")

# %%
ranking = rank_experiments(df)
ranking.head(15)

# %% [markdown]
# ## Pick strategies to compare

# %%
PICK = [
    "BuyAndHold(ticker=QQQ)",
    "BuyAndHold(ticker=NVDA)",
    ranking.iloc[0]["strategy"],   # top-ranked
]
curves = {}
for rec in ok.to_dict("records"):
    if rec["strategy"] in PICK:
        c = registry.load_curve(rec["id"])
        if c is not None:
            curves[rec["strategy"]] = c["returns"]

metrics = pd.DataFrame({k: summary(v) for k, v in curves.items()}).T
metrics[["cagr", "sharpe", "max_drawdown", "calmar", "ann_vol"]]

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 5))
for name, r in curves.items():
    (1 + r).cumprod().plot(ax=ax, label=name[:40], logy=True)
ax.legend()
ax.set_title("Growth of $1 (log)")
plt.show()
