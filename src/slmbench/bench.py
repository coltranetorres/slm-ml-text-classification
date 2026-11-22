"""Latency/throughput/memory benchmarking, run in an isolated subprocess per model."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence

import numpy as np


def summarize_latencies(ms: Sequence[float]) -> tuple[float, float]:
    """(p50, p95) of per-call latencies in milliseconds."""
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))


def run_isolated(
    model_spec: dict,
    train_path: str,
    test_path: str,
    warmup: int = 5,
    n_latency: int = 200,
) -> dict:
    """Fit + benchmark one model in a fresh process so peak RSS is that model's alone.

    Parquet inputs need columns `text` and `label`. Returns accuracy, macro_f1, p50_ms, p95_ms,
    throughput_per_s, peak_rss_mb, peak_mlx_mb.
    """
    args = [json.dumps(model_spec), train_path, test_path, str(warmup), str(n_latency)]
    proc = subprocess.run(
        [sys.executable, "-m", "slmbench.bench_worker", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bench worker failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])
