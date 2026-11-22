import pytest

from slmbench.metrics import INVALID, evaluate


def test_perfect_predictions():
    r = evaluate(["a", "b"], ["a", "b"])
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0


def test_invalid_prediction_counts_wrong():
    r = evaluate(["a", "b"], ["a", INVALID])
    assert r["accuracy"] == 0.5
    assert r["macro_f1"] < 1.0


def test_label_missing_from_predictions_lowers_macro_f1():
    r = evaluate(["a", "a", "b", "b"], ["a", "a", "a", "a"])
    assert r["macro_f1"] == pytest.approx((2 * 0.5 / 1.5 + 0.0) / 2)
