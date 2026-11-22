import hashlib

import numpy as np
import pytest

from slmbench.embeddings import Embedder
from slmbench.models import build_embed_xgb, build_tfidf_lr, build_tfidf_svm

TEXTS = [
    "refund my invoice payment",
    "invoice charged twice billing",
    "billing payment refund",
    "server down outage error",
    "outage network error crash",
    "crash error server outage",
] * 3
LABELS = ["billing"] * 3 + ["tech"] * 3
LABELS = LABELS * 3


def fake_encode(texts):
    out = []
    for t in texts:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % 2**32)
        v = rng.normal(size=8) * 0.01
        v[0] += 1.0 if "billing" in t or "invoice" in t or "refund" in t else -1.0
        out.append(v)
    return np.array(out, dtype=np.float32)


@pytest.fixture
def embedder(tmp_path):
    return Embedder(encode_fn=fake_encode, cache_dir=tmp_path, model_id="fake")


def all_models(embedder):
    return [build_tfidf_lr(), build_tfidf_svm(), build_embed_xgb(embedder, n_estimators=20)]


def test_each_model_fits_and_predicts_known_labels(embedder):
    for m in all_models(embedder):
        m.fit(TEXTS, LABELS)
        assert m.predict(["invoice refund billing", "server outage crash"]) == ["billing", "tech"]


def test_empty_string_prediction_does_not_raise(embedder):
    for m in all_models(embedder):
        m.fit(TEXTS, LABELS)
        assert m.predict([""])[0] in {"billing", "tech"}


def test_non_ascii_text_ok(embedder):
    for m in all_models(embedder):
        m.fit(TEXTS, LABELS)
        assert len(m.predict(["Rückerstattung für Rechnung 😀 日本語"])) == 1


def test_xgb_handles_noncontiguous_label_subset(embedder):
    texts = ["billing refund invoice"] * 5 + ["server outage crash"] * 5
    labels = ["Z-last"] * 5 + ["A-first"] * 5  # unsorted, non-numeric
    m = build_embed_xgb(embedder, n_estimators=10)
    m.fit(texts, labels)
    assert set(m.predict(texts)) <= {"Z-last", "A-first"}


def test_embedder_cache_hits_disk(tmp_path):
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return fake_encode(texts)

    e = Embedder(encode_fn=enc, cache_dir=tmp_path, model_id="fake")
    a = e.embed(["x", "y"])
    b = e.embed(["x", "y"])
    assert calls == [2]
    assert np.allclose(a, b)


def test_embedder_use_cache_false_always_encodes(tmp_path):
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return fake_encode(texts)

    e = Embedder(encode_fn=enc, cache_dir=tmp_path, model_id="fake")
    e.embed(["x"])
    e.use_cache = False
    e.embed(["x"])
    assert calls == [1, 1]
    assert len(list(tmp_path.glob("*.npy"))) == 1  # uncached call wrote nothing new
