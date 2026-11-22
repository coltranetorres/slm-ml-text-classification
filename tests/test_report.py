import pandas as pd

from slmbench.report import make_summary, plot_crossover


def raw_frame() -> pd.DataFrame:
    rows = []
    for gran in ("coarse", "fine"):
        for model in ("tfidf-lr", "tfidf-svm", "embed-xgb"):
            for n in (50, 200):
                for seed in (0, 1):
                    rows.append(
                        (model, gran, n, seed, 0.5, 0.4 + n / 1000, 1.0, 2.0, 50.0, 300.0, 0.0)
                    )
        rows.append(("gemma-3-1b-zeroshot", gran, 0, 0, 0.6, 0.45, 50.0, 70.0, 20.0, 400.0, 2000.0))
    cols = [
        "model",
        "granularity",
        "n_train",
        "seed",
        "accuracy",
        "macro_f1",
        "p50_ms",
        "p95_ms",
        "throughput_per_s",
        "peak_rss_mb",
        "peak_mlx_mb",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_summary_has_one_row_per_model_granularity_size():
    s = make_summary(raw_frame())
    assert not s.duplicated(["model", "granularity", "n_train"]).any()
    assert len(s) == 2 * (3 * 2 + 1)


def test_plot_crossover_writes_png(tmp_path):
    out = tmp_path / "c.png"
    plot_crossover(make_summary(raw_frame()), out)
    assert out.exists() and out.stat().st_size > 1000


def test_majority_baseline():
    from slmbench.report import majority_baseline

    train = pd.Series(["a", "a", "b"])
    test = pd.Series(["a", "b", "b", "b"])
    r = majority_baseline(train, test)
    assert r["accuracy"] == 0.25  # always predicts "a"


def test_plot_handles_two_zeroshot_models(tmp_path):
    raw = raw_frame()
    extra = raw[raw.model == "gemma-3-1b-zeroshot"].copy()
    extra["model"] = "gemma-3-4b-api-zeroshot"
    extra["macro_f1"] = 0.55
    s = make_summary(pd.concat([raw, extra]))
    out = tmp_path / "c.png"
    plotted = plot_crossover(s, out)
    assert out.exists()
    assert "gemma-3-4b-api-zeroshot" in plotted and "gemma-3-1b-zeroshot" in plotted
