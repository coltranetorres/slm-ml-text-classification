import pytest

from slmbench.metrics import INVALID
from slmbench.openrouter import OpenRouterClassifier, parse_label

LABELS = ["Billing", "Billing and Payments", "Technical Support", "IT Support"]


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("Technical Support", "Technical Support"),
        ("  technical support.\n", "Technical Support"),
        ('"IT Support"', "IT Support"),
        ("The category is Technical Support.", "Technical Support"),
        ("Billing and Payments", "Billing and Payments"),  # not shadowed by the "Billing" label
        ("Billing", "Billing"),
        ("It could be Technical Support or IT Support", INVALID),  # ambiguous
        ("I do not know", INVALID),
        ("", INVALID),
    ],
)
def test_parse_label(reply, expected):
    assert parse_label(reply, LABELS) == expected


def make(complete, tmp_path, **kw):
    return OpenRouterClassifier(
        LABELS, model="m/x", complete=complete, cache_dir=tmp_path, backoff_s=0.0, **kw
    )


def test_predict_uses_cache_on_second_call(tmp_path):
    calls = []

    def complete(prompt):
        calls.append(prompt)
        return "Technical Support"

    clf = make(complete, tmp_path)
    assert clf.predict(["my app crashes"]) == ["Technical Support"]
    clf2 = make(complete, tmp_path)  # fresh instance reads the on-disk cache
    assert clf2.predict(["my app crashes"]) == ["Technical Support"]
    assert len(calls) == 1


def test_retries_then_succeeds(tmp_path):
    state = {"n": 0}

    def complete(prompt):
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("429")
        return "IT Support"

    assert make(complete, tmp_path, max_retries=5).predict(["vpn down"]) == ["IT Support"]


def test_raises_after_max_retries_instead_of_scoring_wrong(tmp_path):
    def complete(prompt):
        raise RuntimeError("500")

    with pytest.raises(RuntimeError):
        make(complete, tmp_path, max_retries=2).predict(["x"])


def test_empty_ticket_and_unicode_do_not_raise(tmp_path):
    clf = make(lambda p: "Billing", tmp_path)
    assert clf.predict(["", "Rückerstattung 😀 日本語"]) == ["Billing", "Billing"]


def test_predict_preserves_order_with_concurrency(tmp_path):
    clf = make(lambda p: "IT Support" if "alpha" in p else "Billing", tmp_path, workers=4)
    texts = [f"alpha {i}" if i % 2 == 0 else f"beta {i}" for i in range(20)]
    assert clf.predict(texts) == ["IT Support" if i % 2 == 0 else "Billing" for i in range(20)]
