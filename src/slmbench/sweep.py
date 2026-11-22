"""Training-size x label-granularity sweep over all models."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from slmbench.bench import run_isolated
from slmbench.data import Splits, load_tickets, make_splits, subsample
from slmbench.metrics import evaluate
from slmbench.models import build_tfidf_lr, build_tfidf_svm

log = logging.getLogger(__name__)

C_GRID = (0.1, 1.0, 10.0)
SMALL_N = 200  # sizes <= SMALL_N use every seed; larger sizes use only the first seed
KEY = ["model", "granularity", "n_train", "seed"]
TRAINED = {
    "tfidf-lr": ("tfidf_lr", build_tfidf_lr),
    "tfidf-svm": ("tfidf_svm", build_tfidf_svm),
    "embed-xgb": ("embed_xgb", None),
}
ZEROSHOT = "gemma-3-1b-zeroshot"


def tune_C(builder: Callable, train: pd.DataFrame, val: pd.DataFrame, col: str) -> float:
    """Pick C from C_GRID by validation macro F1 (never touches the test set)."""
    best_c, best = C_GRID[1], -1.0
    for c in C_GRID:
        model = builder(c)
        model.fit(train.text.tolist(), train[col].tolist())
        f1 = evaluate(val[col].tolist(), model.predict(val.text.tolist()))["macro_f1"]
        if f1 > best:
            best_c, best = c, f1
    return best_c


def _write(df: pd.DataFrame, col: str, path: Path) -> str:
    df[["text", col]].rename(columns={col: "label"}).to_parquet(path)
    return str(path)


def run_sweep(
    sizes: Sequence[int],
    seeds: Sequence[int],
    granularities: Sequence[str],
    out_csv: Path,
    df: pd.DataFrame | None = None,
    runner: Callable[..., dict] = run_isolated,
    workdir: Path = Path("data/sweep"),
) -> pd.DataFrame:
    """Run every (model, granularity, size, seed) once; resumable via out_csv."""
    out_csv, workdir = Path(out_csv), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {tuple(r[k] for k in KEY) for r in rows}

    def record(model: str, gran: str, n: int, seed: int, res: dict) -> None:
        rows.append({"model": model, "granularity": gran, "n_train": n, "seed": seed, **res})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log.info("%s %s n=%d seed=%d f1=%.3f", model, gran, n, seed, res["macro_f1"])

    df = load_tickets() if df is None else df
    splits: Splits = make_splits(df, seed=0)
    for gran in granularities:
        col = f"label_{gran}"
        test_path = _write(splits.test, col, workdir / f"test_{gran}.parquet")

        if (ZEROSHOT, gran, 0, 0) not in done:
            train_path = _write(splits.train.head(1), col, workdir / f"zs_train_{gran}.parquet")
            res = runner({"kind": "zeroshot"}, train_path, test_path)
            record(ZEROSHOT, gran, 0, 0, res)

        for size in sizes:
            for seed in seeds if size <= SMALL_N else seeds[:1]:
                sub = subsample(splits.train, size, seed)
                n = len(sub)
                train_path = _write(sub, col, workdir / f"train_{gran}_{n}_{seed}.parquet")
                for name, (kind, builder) in TRAINED.items():
                    if (name, gran, n, seed) in done:
                        continue
                    spec: dict = {"kind": kind}
                    if builder is not None:
                        spec["C"] = tune_C(builder, sub, splits.val, col)
                    record(name, gran, n, seed, runner(spec, train_path, test_path))
                    done.add((name, gran, n, seed))
    return pd.DataFrame(rows)


def run_zeroshot(
    model_id: str,
    name: str,
    granularities: Sequence[str],
    out_csv: Path,
    df: pd.DataFrame | None = None,
    runner: Callable[..., dict] = run_isolated,
    workdir: Path = Path("data/sweep"),
) -> pd.DataFrame:
    """Run only one zero-shot SLM (once per granularity) and append its rows to out_csv.

    Uses the same split and test set as run_sweep, so rows are directly comparable.
    """
    out_csv, workdir = Path(out_csv), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {tuple(r[k] for k in KEY) for r in rows}
    splits = make_splits(load_tickets() if df is None else df, seed=0)
    for gran in granularities:
        if (name, gran, 0, 0) in done:
            continue
        col = f"label_{gran}"
        test_path = _write(splits.test, col, workdir / f"test_{gran}.parquet")
        train_path = _write(splits.train.head(1), col, workdir / f"zs_train_{gran}.parquet")
        res = runner({"kind": "zeroshot", "model_id": model_id}, train_path, test_path)
        rows.append({"model": name, "granularity": gran, "n_train": 0, "seed": 0, **res})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log.info("%s %s f1=%.3f", name, gran, res["macro_f1"])
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds per (model, granularity, n_train)."""
    metrics = [
        c
        for c in (
            "accuracy",
            "macro_f1",
            "p50_ms",
            "p95_ms",
            "throughput_per_s",
            "peak_rss_mb",
            "peak_mlx_mb",
        )
        if c in raw
    ]
    agg = raw.groupby(["model", "granularity", "n_train"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()
