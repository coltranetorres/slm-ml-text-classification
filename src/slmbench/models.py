"""Classical text classifiers behind a common fit/predict protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from slmbench.embeddings import Embedder


class Classifier(Protocol):
    name: str

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None: ...

    def predict(self, texts: Sequence[str]) -> list[str]: ...


class _TfidfClassifier:
    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self._pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1), estimator
        )

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        self._pipe.fit(list(texts), list(labels))

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [str(p) for p in self._pipe.predict(list(texts))]


def build_tfidf_lr(C: float = 1.0) -> Classifier:
    return _TfidfClassifier("tfidf-lr", LogisticRegression(C=C, max_iter=1000))


def build_tfidf_svm(C: float = 1.0) -> Classifier:
    return _TfidfClassifier("tfidf-svm", LinearSVC(C=C))


class _EmbedXGB:
    name = "embed-xgb"

    def __init__(self, embedder: Embedder, **xgb_params) -> None:
        params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "tree_method": "hist"}
        params.update(xgb_params)
        self._embedder = embedder
        self._params = params
        self._le = LabelEncoder()
        self._clf = XGBClassifier(**params)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        y = self._le.fit_transform(list(labels))
        self._clf.fit(self._embedder.embed(list(texts)), y)

    def predict(self, texts: Sequence[str]) -> list[str]:
        pred = np.asarray(self._clf.predict(self._embedder.embed(list(texts)))).astype(int)
        return [str(c) for c in self._le.inverse_transform(pred)]


def build_embed_xgb(embedder: Embedder, **xgb_params) -> Classifier:
    return _EmbedXGB(embedder, **xgb_params)
