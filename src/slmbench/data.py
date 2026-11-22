"""Ticket dataset loading, stratified splits and nested subsampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

DATASET_ID = "Tobi-Bueck/customer-support-tickets"
CACHE_PATH = Path("data/tickets.parquet")
MIN_CLASS_COUNT = 10


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def clean_tickets(raw: pd.DataFrame, min_class_count: int = MIN_CLASS_COUNT) -> pd.DataFrame:
    """English tickets as text/label_coarse/label_fine, exact-duplicate texts removed.

    Duplicates must go before splitting: otherwise the same ticket lands in train and test and
    inflates every memorising model (measured: ~23% of test rows had a train twin).
    """
    raw = raw[raw.language == "en"]
    df = pd.DataFrame(
        {
            "text": (raw.subject.fillna("") + "\n" + raw.body.fillna("")).str.strip(),
            "label_coarse": raw["type"],
            "label_fine": raw["queue"],
        }
    ).dropna()
    df = df.drop_duplicates("text")
    df = df[df.groupby("label_fine")["label_fine"].transform("size") >= min_class_count]
    return df.reset_index(drop=True)


def load_tickets(cache: Path = CACHE_PATH) -> pd.DataFrame:
    """English tickets with columns text, label_coarse (type), label_fine (queue)."""
    if cache.exists():
        return pd.read_parquet(cache)
    from datasets import load_dataset

    df = clean_tickets(load_dataset(DATASET_ID)["train"].to_pandas())
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    log.info("cached %d tickets to %s", len(df), cache)
    return df


def make_splits(
    df: pd.DataFrame, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15
) -> Splits:
    """Deterministic split stratified on label_fine."""
    rest, test = train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=df.label_fine
    )
    train, val = train_test_split(
        rest, test_size=val_frac / (1 - test_frac), random_state=seed, stratify=rest.label_fine
    )
    return Splits(train=train, val=val, test=test)


def subsample(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Class-proportional subsample of size n; nested across n for a fixed seed.

    Each row gets key (rank_in_class + u) / class_size with u ~ U[0,1) fixed by the seed, so
    the n smallest keys are a prefix of the n' > n smallest. Never raises.
    """
    if n >= len(train):
        return train
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    shuffled = train.iloc[order]
    rank = shuffled.groupby("label_fine").cumcount().to_numpy()
    size = shuffled.groupby("label_fine")["label_fine"].transform("size").to_numpy()
    key = (rank + rng.random(len(train))) / size
    return shuffled.iloc[np.argsort(key, kind="stable")[:n]]
