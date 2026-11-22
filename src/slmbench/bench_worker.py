"""Subprocess entry point: python -m slmbench.bench_worker SPEC TRAIN TEST WARMUP N_LATENCY."""

from __future__ import annotations

import json
import resource
import sys
import time

import pandas as pd

from slmbench.bench import summarize_latencies
from slmbench.metrics import evaluate
from slmbench.models import Classifier


def build_from_spec(spec: dict, labels: list[str]) -> Classifier:
    kind = spec["kind"]
    if kind == "tfidf_lr":
        from slmbench.models import build_tfidf_lr

        return build_tfidf_lr(spec.get("C", 1.0))
    if kind == "tfidf_svm":
        from slmbench.models import build_tfidf_svm

        return build_tfidf_svm(spec.get("C", 1.0))
    if kind == "embed_xgb":
        from slmbench.embeddings import Embedder
        from slmbench.models import build_embed_xgb

        return build_embed_xgb(Embedder(), **spec.get("xgb", {}))
    if kind == "zeroshot":
        from slmbench.zeroshot import MODEL_ID, ZeroShotClassifier

        return ZeroShotClassifier(labels, model_id=spec.get("model_id", MODEL_ID))
    raise ValueError(f"unknown model kind: {kind}")


def main(argv: list[str]) -> None:
    spec, train_path, test_path = json.loads(argv[0]), argv[1], argv[2]
    warmup, n_latency = int(argv[3]), int(argv[4])
    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    labels = sorted(set(train.label) | set(test.label))
    model = build_from_spec(spec, labels)
    model.fit(train.text.tolist(), train.label.tolist())

    texts = test.text.tolist()
    embedder = getattr(model, "embedder", None)
    if embedder is not None:
        embedder.use_cache = False  # time real encoding, not disk hits
    probe = texts[:n_latency]
    for t in probe[:warmup] or texts[:1]:
        model.predict([t])
    lat_ms = []
    for t in probe:
        s = time.perf_counter()
        model.predict([t])
        lat_ms.append((time.perf_counter() - s) * 1000)
    s = time.perf_counter()
    model.predict(probe)
    throughput = len(probe) / (time.perf_counter() - s)

    if embedder is not None:
        embedder.use_cache = True
    metrics = evaluate(test.label.tolist(), model.predict(texts))
    p50, p95 = summarize_latencies(lat_ms)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS
    try:
        import mlx.core as mx

        mlx_mb = mx.get_peak_memory() / 1e6 if spec["kind"] == "zeroshot" else 0.0
    except Exception:
        mlx_mb = 0.0
    print(
        json.dumps(
            {
                **metrics,
                "p50_ms": p50,
                "p95_ms": p95,
                "throughput_per_s": throughput,
                "peak_rss_mb": rss_mb,
                "peak_mlx_mb": mlx_mb,
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
