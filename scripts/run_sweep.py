"""Run the full sweep (long). Resumable: re-run to continue after an interruption."""

import logging
from pathlib import Path

from slmbench.sweep import run_sweep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    run_sweep(
        sizes=[50, 200, 1000, 10**9],  # 10**9 = full training set
        seeds=[0, 1, 2],
        granularities=["coarse", "fine"],
        out_csv=Path("results/raw.csv"),
    )
