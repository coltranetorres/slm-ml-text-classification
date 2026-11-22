"""Summary table and crossover chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from slmbench.sweep import ZEROSHOT, aggregate  # noqa: E402

# Fixed categorical order (validated light palette, slots 1-4). Contrast vs surface is < 3:1 for
# two hues, so every series also gets a distinct marker, a legend and direct labels.
STYLE = {
    "tfidf-lr": ("#2a78d6", "o", "TF-IDF + LogReg"),
    "tfidf-svm": ("#eb6834", "s", "TF-IDF + SVM"),
    "embed-xgb": ("#1baf7a", "^", "MiniLM + XGBoost"),
    ZEROSHOT: ("#6250d6", "D", "Gemma-3-1B zero-shot"),
    "gemma-3-4b-zeroshot": ("#e87ba4", "D", "Gemma-3-4B zero-shot"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TITLE = {
    "coarse": "Coarse labels: ticket type (4 classes)",
    "fine": "Fine labels: queue (10 classes)",
}


def make_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, granularity, n_train) with mean/std over seeds."""
    return aggregate(raw).round(4)


def plot_crossover(summary: pd.DataFrame, out: Path) -> None:
    grans = [g for g in ("coarse", "fine") if g in set(summary.granularity)]
    fig, axes = plt.subplots(1, len(grans), figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    axes = [axes] if len(grans) == 1 else list(axes)
    for ax, gran in zip(axes, grans, strict=True):
        ax.set_facecolor(SURFACE)
        part = summary[summary.granularity == gran]
        for model, (color, marker, label) in STYLE.items():
            m = part[part.model == model].sort_values("n_train")
            if m.empty:
                continue
            if model.endswith("-zeroshot"):
                y = m.macro_f1_mean.iloc[0]
                ax.axhline(y, color=color, lw=2, ls="--", label=label)
                continue
            ax.errorbar(
                m.n_train,
                m.macro_f1_mean,
                yerr=m.macro_f1_std.fillna(0),
                color=color,
                marker=marker,
                ms=7,
                lw=2,
                capsize=3,
                label=label,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
        ax.set_xscale("log")
        ax.set_title(TITLE[gran], loc="left", color=INK, fontsize=11)
        ax.set_xlabel("training tickets (log scale)", color=MUTED)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("macro F1 (test set)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def majority_baseline(train_labels: pd.Series, test_labels: pd.Series) -> dict[str, float]:
    """Accuracy/macro F1 of always predicting the most common training label."""
    from slmbench.metrics import evaluate

    top = train_labels.value_counts().idxmax()
    return evaluate(test_labels.tolist(), [top] * len(test_labels))
