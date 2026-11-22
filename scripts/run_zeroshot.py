"""Run one extra zero-shot SLM only (classical results in results/raw.csv are kept).

Example: uv run python scripts/run_zeroshot.py mlx-community/gemma-3-4b-it-bf16 gemma-3-4b-zeroshot
"""

import argparse
import logging
from pathlib import Path

from slmbench.sweep import run_zeroshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("name", help="row name in raw.csv, e.g. gemma-3-4b-zeroshot")
    args = ap.parse_args()
    run_zeroshot(args.model_id, args.name, ["coarse", "fine"], Path("results/raw.csv"))
