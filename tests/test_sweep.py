import pandas as pd

from slmbench.sweep import aggregate, run_sweep


def toy(n_per_class: int = 40, classes: int = 4) -> pd.DataFrame:
    rows = [
        {"text": f"ticket {c} {i}", "label_coarse": f"c{c % 2}", "label_fine": f"f{c}"}
        for c in range(classes)
        for i in range(n_per_class)
    ]
    return pd.DataFrame(rows)


def make_stub():
    calls = []

    def stub(spec, train_path, test_path, **kw):
        calls.append(spec["kind"])
        return {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "p50_ms": 1.0,
            "p95_ms": 2.0,
            "throughput_per_s": 100.0,
            "peak_rss_mb": 300.0,
            "peak_mlx_mb": 0.0,
        }

    return stub, calls


def sweep(tmp_path, stub, **kw):
    return run_sweep(
        sizes=[10, 40],
        seeds=[0, 1],
        granularities=["coarse", "fine"],
        out_csv=tmp_path / "raw.csv",
        df=toy(),
        runner=stub,
        workdir=tmp_path / "w",
        **kw,
    )


def test_sweep_writes_expected_row_count(tmp_path):
    stub, _ = make_stub()
    out = sweep(tmp_path, stub)
    # per granularity: 3 trained models x (size 10 x 2 seeds + size 40 x 1 seed... ) + 1 zeroshot
    # small-size threshold is 200, so both sizes use both seeds: 3 x 2 x 2 + 1
    assert len(out) == 2 * (3 * 2 * 2 + 1)


def test_sweep_is_resumable(tmp_path):
    stub, calls = make_stub()
    sweep(tmp_path, stub)
    n = len(calls)
    out = sweep(tmp_path, stub)
    assert len(calls) == n and len(out) == 2 * (3 * 2 * 2 + 1)


def test_zeroshot_runs_once_per_granularity(tmp_path):
    stub, calls = make_stub()
    sweep(tmp_path, stub)
    assert calls.count("zeroshot") == 2


def test_aggregate_mean_std():
    raw = pd.DataFrame(
        {
            "model": ["m", "m"],
            "granularity": ["fine"] * 2,
            "n_train": [50, 50],
            "seed": [0, 1],
            "accuracy": [0.4, 0.6],
            "macro_f1": [0.2, 0.4],
            "p50_ms": [1.0, 1.0],
            "p95_ms": [2.0, 2.0],
            "throughput_per_s": [10.0, 10.0],
            "peak_rss_mb": [5.0, 5.0],
        }
    )
    agg = aggregate(raw)
    row = agg.iloc[0]
    assert abs(row.macro_f1_mean - 0.3) < 1e-9 and abs(row.macro_f1_std - 0.1414213562) < 1e-6
