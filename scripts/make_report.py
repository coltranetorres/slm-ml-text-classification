"""Build results/summary.csv, results/baselines.csv and results/crossover.png from raw.csv."""

from pathlib import Path

import pandas as pd

from slmbench.data import load_tickets, make_splits
from slmbench.report import majority_baseline, make_summary, plot_crossover

if __name__ == "__main__":
    summary = make_summary(pd.read_csv("results/raw.csv"))
    summary.to_csv("results/summary.csv", index=False)
    plot_crossover(summary, Path("results/crossover.png"))
    splits = make_splits(load_tickets(), seed=0)
    base = pd.DataFrame(
        [
            {
                "granularity": g,
                **majority_baseline(splits.train[f"label_{g}"], splits.test[f"label_{g}"]),
            }
            for g in ("coarse", "fine")
        ]
    )
    base.to_csv("results/baselines.csv", index=False)
    print(summary.to_string())
    print(base.to_string())
