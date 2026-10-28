"""Utility functions for reproducibility, evaluation metrics, and visualization."""

import os
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data import ID2LABEL, CLASS_NAMES


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
