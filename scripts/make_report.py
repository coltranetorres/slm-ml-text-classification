"""Build results/summary.csv and results/crossover.png from results/raw.csv."""

from pathlib import Path

import pandas as pd

from slmbench.report import make_summary, plot_crossover

if __name__ == "__main__":
    summary = make_summary(pd.read_csv("results/raw.csv"))
    summary.to_csv("results/summary.csv", index=False)
    plot_crossover(summary, Path("results/crossover.png"))
    print(summary.to_string())
