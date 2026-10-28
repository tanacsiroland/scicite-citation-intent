"""Metric computation and evaluation routines for classification tasks."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics.

    Calculates overall Accuracy, Macro, Micro, and Weighted averages
    (Precision, Recall, F1-score) along with class-level decompositions
    and the confusion matrix.

    Args:
        y_true: Ground truth target labels (integers).
        y_pred: Predicted target labels (integers).
        id2label: Dictionary mapping label ID to human-readable string.

    Returns:
        Dictionary containing overall metrics, per-class breakdown, and confusion matrix.
    """
    labels = sorted(id2label.keys())
    target_names = [id2label[i] for i in labels]

    # Accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Macro averages (unweighted arithmetic mean across classes)
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Micro averages (globally aggregated true positives, false negatives, false positives)
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    # Weighted averages (support-weighted mean across classes)
    weighted_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    p_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    r_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    per_class_records = []
    for idx, label_id in enumerate(labels):
        support = int((np.array(y_true) == label_id).sum())
        per_class_records.append({
            "label_id": label_id,
            "label_name": id2label[label_id],
            "precision": round(float(p_per_class[idx]), 4),
            "recall": round(float(r_per_class[idx]), 4),
            "f1": round(float(f1_per_class[idx]), 4),
            "support": support,
        })

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(acc, 4),
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
        },
        "weighted": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1": round(weighted_f1, 4),
        },
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "per_class": per_class_records,
        "confusion_matrix": cm.tolist(),
        "target_names": target_names,
    }


def metrics_to_dataframe(metrics_dict: Dict[str, Any]) -> pd.DataFrame:
    """Format per-class metrics and all aggregate averages into a presentation DataFrame.

    Includes individual classes followed by accuracy, micro avg, macro avg, and weighted avg.

    Args:
        metrics_dict: Output dictionary from compute_metrics.

    Returns:
        DataFrame containing complete classification report data.
    """
    class_df = pd.DataFrame(metrics_dict["per_class"])
    total_support = sum(r["support"] for r in metrics_dict["per_class"])

    summary_rows = [
        {
            "label_id": -1,
            "label_name": "accuracy",
            "precision": metrics_dict["accuracy"],
            "recall": metrics_dict["accuracy"],
            "f1": metrics_dict["accuracy"],
            "support": total_support,
        },
        {
            "label_id": -2,
            "label_name": "micro avg",
            "precision": metrics_dict["micro_precision"],
            "recall": metrics_dict["micro_recall"],
            "f1": metrics_dict["micro_f1"],
            "support": total_support,
        },
        {
            "label_id": -3,
            "label_name": "macro avg",
            "precision": metrics_dict["macro_precision"],
            "recall": metrics_dict["macro_recall"],
            "f1": metrics_dict["macro_f1"],
            "support": total_support,
        },
        {
            "label_id": -4,
            "label_name": "weighted avg",
            "precision": metrics_dict["weighted_precision"],
            "recall": metrics_dict["weighted_recall"],
            "f1": metrics_dict["weighted_f1"],
            "support": total_support,
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    return pd.concat([class_df, summary_df], ignore_index=True)
