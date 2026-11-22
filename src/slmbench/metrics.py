"""Metrics and the per-run result schema."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score

INVALID = "__invalid__"


@dataclass
class ResultRow:
    model: str
    granularity: str
    n_train: int
    seed: int
    accuracy: float
    macro_f1: float
    p50_ms: float
    p95_ms: float
    throughput_per_s: float
    peak_rss_mb: float


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Accuracy and macro F1 over the union of true and predicted labels.

    Invalid predictions are never dropped; they count as wrong (and add a zero-F1 class
    only when they appear, which is the desired penalty).
    """
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }
