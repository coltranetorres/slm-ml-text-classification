import pandas as pd
import pytest

from slmbench.bench import run_isolated, summarize_latencies


def test_summarize_latencies_known_values():
    p50, p95 = summarize_latencies([float(i) for i in range(1, 101)])
    assert p50 == pytest.approx(50.5)
    assert p95 == pytest.approx(95.05)


def test_run_isolated_returns_all_keys(tmp_path):
    rows = [("refund invoice billing", "billing"), ("server outage crash", "tech")] * 10
    df = pd.DataFrame(rows, columns=["text", "label"])
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    df.to_parquet(train)
    df.to_parquet(test)
    out = run_isolated({"kind": "tfidf_lr", "C": 1.0}, str(train), str(test), warmup=1, n_latency=5)
    assert set(out) >= {
        "accuracy",
        "macro_f1",
        "p50_ms",
        "p95_ms",
        "throughput_per_s",
        "peak_rss_mb",
        "peak_mlx_mb",
    }
    assert out["accuracy"] == 1.0
    assert out["peak_rss_mb"] > 0
