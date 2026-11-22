"""Score google/gemma-3-4b-it zero-shot via OpenRouter on the shared test set (appends to raw.csv).

Needs OPENROUTER_API_KEY in .env (git-ignored). ~8.5k calls, roughly $0.10 at $0.04/$0.05 per 1M.
Responses are cached in data/cache/, so an interrupted run resumes without re-paying.
"""

import logging
from pathlib import Path

from slmbench.openrouter import MODEL, OpenRouterClassifier
from slmbench.sweep import run_api_zeroshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

if __name__ == "__main__":
    run_api_zeroshot(
        lambda labels: OpenRouterClassifier(labels, model=MODEL),
        name="gemma-3-4b-api-zeroshot",
        granularities=["coarse", "fine"],
        out_csv=Path("results/raw.csv"),
    )
