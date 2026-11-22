"""Recon of candidate HF ticket datasets: size, labels, text length, quick TF-IDF+LR baseline."""

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def baseline(df: pd.DataFrame, label: str) -> float:
    d = df.dropna(subset=["text", label])
    d = d[d.groupby(label)[label].transform("size") >= 10]
    tr, te = train_test_split(d, test_size=0.2, random_state=0, stratify=d[label])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(tr.text), tr[label])
    return f1_score(te[label], clf.predict(vec.transform(te.text)), average="macro")


def report(name: str, df: pd.DataFrame, labels: list[str]) -> None:
    print(f"\n== {name}: rows={len(df)} mean_chars={df.text.str.len().mean():.0f}")
    for lab in labels:
        vc = df[lab].value_counts()
        print(
            f"  [{lab}] classes={len(vc)} top/bottom={vc.iloc[0]}/{vc.iloc[-1]} "
            f"TFIDF-LR macroF1={baseline(df, lab):.3f}"
        )


def main() -> None:
    t = load_dataset("Tobi-Bueck/customer-support-tickets")["train"].to_pandas()
    t = t[t.language == "en"].copy()
    t["text"] = t.subject.fillna("") + "\n" + t.body.fillna("")
    report("Tobi-Bueck (en)", t, ["queue", "type", "priority", "tag_1"])

    ds = load_dataset("tasksource/it-support-tickets")
    names = ds["train"].features["label"].names
    it = pd.concat([ds[s].to_pandas() for s in ds])
    it["label"] = it.label.map(dict(enumerate(names)))
    report("tasksource/it-support-tickets", it, ["label"])

    b = load_dataset("PolyAI/banking77")["train"].to_pandas()
    b["label"] = b.label.astype(str)
    report("banking77", b, ["label"])


if __name__ == "__main__":
    main()
