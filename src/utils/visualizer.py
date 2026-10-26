"""Visualization utilities for EDA and model evaluation."""

from typing import Dict, List, Optional
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
