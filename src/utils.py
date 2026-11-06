"""Utility functions for reproducibility, evaluation metrics, and visualization."""

import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data import ID2LABEL, CLASS_NAMES


def display_saved_figure(
    image_path: str | os.PathLike,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """Load and display an image file from disk cleanly in Jupyter/Matplotlib."""
    import matplotlib.image as mpimg
    p = Path(image_path)
    if not p.is_file():
        print(f"Warning: Figure not found at {p}")
        return
    plt.figure(figsize=figsize)
    plt.imshow(mpimg.imread(str(p)))
    plt.axis("off")
    if title:
        plt.title(title, fontsize=11, fontweight="bold", pad=8)
    plt.show()


def plot_class_distribution(
    dfs: Dict[str, pd.DataFrame],
    id2label: Optional[Dict[int, str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot label percentage distribution across train, validation, and test splits."""
    if id2label is None:
        id2label = ID2LABEL

    records = []
    for split_name, df in dfs.items():
        counts = df["label"].map(id2label).value_counts(normalize=True) * 100
        for label_name, pct in counts.items():
            records.append({
                "Split": split_name.capitalize(),
                "Citation Intent": label_name.capitalize(),
                "Percentage (%)": round(pct, 2),
                "Count": int((df["label"].map(id2label) == label_name).sum()),
            })

    plot_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = sns.color_palette("muted", n_colors=3)
    sns.barplot(
        data=plot_df,
        x="Citation Intent",
        y="Percentage (%)",
        hue="Split",
        palette=palette,
        ax=ax,
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax.set_title("Citation Intent Class Distribution across Dataset Splits", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_xlabel("Citation Intent", fontsize=10)
    ax.set_ylim(0, max(plot_df["Percentage (%)"]) * 1.15)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_canonical_section_contingency(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot conditional probability P(Intent | Normalized Section) heatmap."""
    contingency = pd.crosstab(df["section_clean"], df["label_name"], normalize="index") * 100
    cols = [c for c in ["background", "method", "result"] if c in contingency.columns]
    contingency = contingency[cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        contingency,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Conditional Probability (%)"},
        ax=ax,
    )
    ax.set_title("Intent Distribution by Normalized Section P(Intent | Section)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Canonical Section Category", fontsize=10)
    ax.set_xlabel("Citation Intent", fontsize=10)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_prediction_probabilities(
    probabilities: Dict[str, float],
    predicted_label: str,
    title: str = "Citation Intent Prediction",
) -> plt.Figure:
    """Horizontal bar chart showing model confidence across the 3 citation classes."""
    classes = list(probabilities.keys())
    scores = [probabilities[c] for c in classes]

    colors = ["#2b5c8f" if c.lower() == predicted_label.lower() else "#b0c4de" for c in classes]

    fig, ax = plt.subplots(figsize=(7, 2.6))
    bars = ax.barh(classes, scores, color=colors, height=0.55)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Softmax Probability", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold" if score == max(scores) else "normal",
        )

    plt.tight_layout()
    return fig


def set_seed(seed: int = 42) -> None:
    """Set random seed across random, numpy, and torch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def compute_classification_metrics(
    y_true: np.ndarray | List[int],
    y_pred: np.ndarray | List[int],
    id2label: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compute overall and per-class classification metrics."""
    if id2label is None:
        id2label = ID2LABEL

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    labels = sorted(id2label.keys())

    acc = float(accuracy_score(y_t, y_p))
    macro_p = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    macro_r = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    p_per_class = precision_score(y_t, y_p, average=None, labels=labels, zero_division=0)
    r_per_class = recall_score(y_t, y_p, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(y_t, y_p, average=None, labels=labels, zero_division=0)

    per_class = {}
    for idx, label_id in enumerate(labels):
        name = id2label[label_id]
        support = int((y_t == label_id).sum())
        per_class[name] = {
            "precision": round(float(p_per_class[idx]), 4),
            "recall": round(float(r_per_class[idx]), 4),
            "f1": round(float(f1_per_class[idx]), 4),
            "support": support,
        }

    cm = confusion_matrix(y_t, y_p, labels=labels)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrices(
    df: pd.DataFrame,
    models: List[Tuple[str, str]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 2x2 comparison grid of row-normalized confusion matrices."""
    labels = [0, 1, 2]
    target_names = [CLASS_NAMES[i].capitalize() for i in labels]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, (title, col) in enumerate(models):
        ax = axes[idx]
        cm = confusion_matrix(df["label"], df[col], labels=labels)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            cbar=True,
            ax=ax,
            vmin=0,
            vmax=1.0,
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

    plt.suptitle("Confusion Matrix Comparison Across Key Architectures (Test Set)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
