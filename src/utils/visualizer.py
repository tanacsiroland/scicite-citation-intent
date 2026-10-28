"""Visualization utilities for EDA and model evaluation."""

from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure publication-grade styling
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


def plot_class_distribution(
    dfs: Dict[str, pd.DataFrame],
    id2label: Dict[int, str],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot label distribution across train, validation, and test splits.

    Args:
        dfs: Mapping of split name to pandas DataFrame.
        id2label: Integer to class name mapping.
        output_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
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

    # Annotate bar percentages
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
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_top_sections(
    df: pd.DataFrame,
    top_n: int = 15,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the most frequent raw section names in the training set.

    Args:
        df: Training DataFrame.
        top_n: Number of top sections to display.
        output_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    # Treat empty strings or None as 'MISSING / UNKNOWN'
    sections = df["sectionName"].fillna("").replace("", "[MISSING / NONE]")
    counts = sections.value_counts().head(top_n).reset_index()
    counts.columns = ["sectionName", "count"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=counts, y="sectionName", x="count", hue="sectionName", palette="Blues_r", legend=False, ax=ax)

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{int(width):,}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=9,
            xytext=(5, 0),
            textcoords="offset points",
        )

    ax.set_title(f"Top {top_n} Most Frequent Raw Section Names (Train Split)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Occurrences", fontsize=10)
    ax.set_ylabel("Raw Section Name", fontsize=10)
    ax.set_xlim(0, max(counts["count"]) * 1.15)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_section_label_contingency(
    df: pd.DataFrame,
    id2label: Dict[int, str],
    top_n: int = 10,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot conditional probability P(label | section) for top sections.

    Args:
        df: Training DataFrame.
        id2label: Integer to class name mapping.
        top_n: Number of top sections to analyze.
        output_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    cleaned = df.copy()
    cleaned["section_display"] = cleaned["sectionName"].fillna("").replace("", "[MISSING]")
    top_sections = cleaned["section_display"].value_counts().head(top_n).index

    subset = cleaned[cleaned["section_display"].isin(top_sections)].copy()
    subset["intent"] = subset["label"].map(id2label)

    contingency = pd.crosstab(subset["section_display"], subset["intent"], normalize="index") * 100
    # Reindex to order by frequency in top_sections
    contingency = contingency.loc[top_sections]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        contingency,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Percentage within Section (%)"},
        ax=ax,
    )

    ax.set_title(f"Intent Distribution by Section Header P(Intent | Section) [Top {top_n}]", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Section Header", fontsize=10)
    ax.set_xlabel("Citation Intent", fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_canonical_section_distribution(
    processed_dfs: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot distribution of normalized canonical section categories across splits.

    Args:
        processed_dfs: Mapping of split name to preprocessed DataFrame.
        output_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    records = []
    for split_name, df in processed_dfs.items():
        pcts = df["section_clean"].value_counts(normalize=True) * 100
        for section_name, pct in pcts.items():
            records.append({
                "Split": split_name.capitalize(),
                "Canonical Section": section_name,
                "Percentage (%)": round(pct, 2),
                "Count": int((df["section_clean"] == section_name).sum()),
            })

    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    palette = sns.color_palette("muted", n_colors=3)
    sns.barplot(
        data=plot_df,
        x="Canonical Section",
        y="Percentage (%)",
        hue="Split",
        palette=palette,
        ax=ax,
    )

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.set_title("Distribution of Normalized Canonical Sections Across Splits", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_xlabel("Canonical Section", fontsize=10)
    ax.set_ylim(0, max(plot_df["Percentage (%)"]) * 1.15)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_canonical_section_contingency(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot conditional probability P(Intent | Canonical Section) heatmap.

    Demonstrates the strong predictive prior established by section normalization.

    Args:
        df: DataFrame containing 'section_clean' and 'label_name'.
        output_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    contingency = pd.crosstab(df["section_clean"], df["label_name"], normalize="index") * 100

    # Ensure consistent column ordering if present
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
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray | List[List[int]],
    target_names: List[str],
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot confusion matrix with count and row-normalized percentages.

    Args:
        cm: Confusion matrix array or list.
        target_names: Class labels in order.
        title: Figure title.
        output_path: Optional path to save figure.
        normalize: Whether to normalize by true labels (row-wise).

    Returns:
        Matplotlib Figure object.
    """
    cm_arr = np.array(cm)
    if normalize:
        cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
        annot = np.empty_like(cm_arr, dtype=object)
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                annot[i, j] = f"{cm_arr[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
        data_to_plot = cm_norm
        fmt = ""
    else:
        data_to_plot = cm_arr
        annot = True
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        data_to_plot,
        annot=annot,
        fmt=fmt,
        cmap="Blues",
        xticklabels=[t.capitalize() for t in target_names],
        yticklabels=[t.capitalize() for t in target_names],
        cbar=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("True Intent Label", fontsize=10)
    ax.set_xlabel("Predicted Intent Label", fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_curves(
    histories: Dict[str, List[Dict[str, Any]]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot multi-epoch training curves structured as a 2x3 small multiples grid.

    Columns: Model architectures (DistilBERT, ModernBERT, SciBERT).
    Rows:
      - Row 0: Validation Loss (shared y-axis, highlighting minimum loss / onset of overfitting).
      - Row 1: Validation Macro-F1 (shared y-axis, highlighting peak epoch and exact metric).

    Args:
        histories: Mapping of run_name to list of log_history records.
        output_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    model_families = [
        ("distilbert", "DistilBERT (66M)"),
        ("modernbert", "ModernBERT (149M)"),
        ("scibert", "SciBERT (110M)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)

    color_text_only = "#1f77b4"      # Steel blue
    color_section_aware = "#d62728"  # Crimson red

    # First pass: determine global min/max for aligned y-limits
    all_losses = []
    all_f1s = []
    for h in histories.values():
        for log in h:
            if "eval_loss" in log:
                all_losses.append(log["eval_loss"])
            if "eval_macro_f1" in log:
                all_f1s.append(log["eval_macro_f1"])

    loss_min = min(all_losses) * 0.9 if all_losses else 0.3
    loss_max = max(all_losses) * 1.05 if all_losses else 1.4
    f1_min = min(all_f1s) * 0.99 if all_f1s else 0.80
    f1_max = max(all_f1s) * 1.01 if all_f1s else 0.86

    for col_idx, (model_prefix, display_title) in enumerate(model_families):
        ax_loss = axes[0, col_idx]
        ax_f1 = axes[1, col_idx]

        rep_configs = [
            (f"{model_prefix}_text_only", "Text-only", color_text_only, "o", "-"),
            (f"{model_prefix}_text_section_aware", "Section-aware", color_section_aware, "s", "--"),
        ]

        for run_key, rep_label, color, marker, linestyle in rep_configs:
            history = histories.get(run_key, [])
            epochs, losses, f1s = [], [], []

            for log in history:
                if "eval_macro_f1" in log and "epoch" in log:
                    epochs.append(int(log["epoch"]))
                    f1s.append(log["eval_macro_f1"])
                    if "eval_loss" in log:
                        losses.append(log["eval_loss"])

            if not epochs:
                continue

            # Plot Loss (Row 0)
            ax_loss.plot(
                epochs,
                losses,
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=rep_label,
                linewidth=2.2,
                markersize=6,
            )

            # Annotate Minimum Loss
            min_loss_val = min(losses)
            best_loss_ep = epochs[losses.index(min_loss_val)]
            ax_loss.scatter([best_loss_ep], [min_loss_val], color=color, s=110, zorder=5)
            y_offset = -20 if rep_label == "Text-only" else 15
            ax_loss.annotate(
                f"Min: {min_loss_val:.3f}\n(Ep {best_loss_ep})",
                (best_loss_ep, min_loss_val),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9, lw=0.8),
            )

            # Plot Macro-F1 (Row 1)
            ax_f1.plot(
                epochs,
                f1s,
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=rep_label,
                linewidth=2.2,
                markersize=6,
            )

            # Annotate Peak Macro-F1
            peak_f1_val = max(f1s)
            peak_f1_ep = epochs[f1s.index(peak_f1_val)]
            ax_f1.scatter([peak_f1_ep], [peak_f1_val], color=color, s=140, marker="*", zorder=5)
            y_offset_f1 = 15 if rep_label == "Section-aware" else -25
            ax_f1.annotate(
                f"Peak: {peak_f1_val:.4f}\n(Ep {peak_f1_ep})",
                (peak_f1_ep, peak_f1_val),
                textcoords="offset points",
                xytext=(0, y_offset_f1),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9, lw=0.8),
            )

        # Style Loss subplots (Row 0)
        ax_loss.set_title(f"{display_title}\nValidation Loss", fontsize=11, fontweight="bold", pad=8)
        ax_loss.set_ylim(loss_min, loss_max)
        ax_loss.set_xticks(range(1, 11))
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        if col_idx == 0:
            ax_loss.set_ylabel("Validation Loss\n(Lower is better)", fontsize=10, fontweight="bold")
        ax_loss.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

        # Style Macro-F1 subplots (Row 1)
        ax_f1.set_title(f"{display_title}\nValidation Macro-F1", fontsize=11, fontweight="bold", pad=8)
        ax_f1.set_ylim(f1_min, f1_max)
        ax_f1.set_xticks(range(1, 11))
        ax_f1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
        ax_f1.grid(True, linestyle="--", alpha=0.5)
        if col_idx == 0:
            ax_f1.set_ylabel("Validation Macro-F1\n(Higher is better)", fontsize=10, fontweight="bold")
        ax_f1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Multi-Epoch Training Trajectory by Model Architecture (10-Epoch Overfitting Analysis)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
