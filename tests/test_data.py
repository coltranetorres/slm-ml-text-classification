import pandas as pd

from slmbench.data import make_splits, subsample


def toy(n_per_class: int = 40, classes: int = 5) -> pd.DataFrame:
    rows = [
        {"text": f"ticket {c} {i}", "label_coarse": f"c{c % 2}", "label_fine": f"f{c}"}
        for c in range(classes)
        for i in range(n_per_class)
    ]
    return pd.DataFrame(rows)


def test_splits_are_disjoint_and_stratified():
    s = make_splits(toy(), seed=0)
    assert not set(s.train.index) & set(s.val.index)
    assert not set(s.train.index) & set(s.test.index)
    assert not set(s.val.index) & set(s.test.index)
    assert len(s.train) + len(s.val) + len(s.test) == 200
    assert set(s.test.label_fine) == {f"f{c}" for c in range(5)}
    assert s.test.label_fine.value_counts().nunique() == 1  # equal classes stay equal


def test_splits_deterministic_for_seed():
    a, b = make_splits(toy(), seed=3), make_splits(toy(), seed=3)
    assert list(a.test.index) == list(b.test.index)


def test_subsample_nested():
    train = make_splits(toy(), seed=0).train
    small, big = subsample(train, 20, seed=1), subsample(train, 60, seed=1)
    assert set(small.index) <= set(big.index)
    assert len(small) == 20 and len(big) == 60


def test_subsample_smaller_than_class_count_does_not_raise():
    train = make_splits(toy(classes=5), seed=0).train
    assert len(subsample(train, 3, seed=0)) == 3


def test_subsample_larger_than_train_returns_all():
    train = make_splits(toy(), seed=0).train
    assert len(subsample(train, 10_000, seed=0)) == len(train)
