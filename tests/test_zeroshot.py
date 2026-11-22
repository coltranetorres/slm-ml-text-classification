from slmbench.metrics import INVALID
from slmbench.zeroshot import build_prompt, pick_label, truncate_text, unique_prefixes

LABELS = ["Billing and Payments", "Technical Support", "IT Support"]


def test_prompt_contains_all_labels_and_ticket():
    p = build_prompt("my invoice is wrong", LABELS)
    assert "my invoice is wrong" in p
    assert all(label in p for label in LABELS)


def test_truncate_text_respects_token_budget():
    enc = lambda s: s.split()  # noqa: E731 - stub tokenizer: one token per word
    dec = lambda toks: " ".join(toks)  # noqa: E731
    out = truncate_text("word " * 10_000, 256, enc, dec)
    assert len(enc(out)) == 256
    assert truncate_text("short ticket", 256, enc, dec) == "short ticket"


def test_pick_label_argmax():
    assert pick_label({"a": -2.0, "b": -0.5, "c": -9.0}) == "b"


def test_pick_label_empty_scores_returns_invalid_sentinel():
    assert pick_label({}) == INVALID


def test_unique_prefixes_extends_until_distinct():
    toks = {"IT Support": [5, 7], "IT Billing": [5, 9], "Sales": [3, 1]}
    pref = unique_prefixes(toks)
    assert pref == {"IT Support": [5, 7], "IT Billing": [5, 9], "Sales": [3]}
