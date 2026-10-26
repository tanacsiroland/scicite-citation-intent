"""Exploratory Data Analysis (EDA) module for the SciCite corpus."""

import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_split_summary(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize overall row counts, columns, and missingness across splits.

    Args:
        dfs: Mapping of split name to DataFrame.

    Returns:
        Summary DataFrame.
    """
    summary = []
    for split_name, df in dfs.items():
        summary.append({
            "split": split_name,
            "total_samples": len(df),
            "unique_citing_papers": df["citingPaperId"].nunique() if "citingPaperId" in df else None,
            "unique_cited_papers": df["citedPaperId"].nunique() if "citedPaperId" in df else None,
            "missing_string": df["string"].isna().sum(),
            "missing_section": df["sectionName"].isna().sum() + (df["sectionName"] == "").sum(),
            "section_missing_pct": round(
                (df["sectionName"].isna().sum() + (df["sectionName"] == "").sum()) / len(df) * 100, 2
            ),
        })
    return pd.DataFrame(summary)


def compute_class_distribution(
    dfs: Dict[str, pd.DataFrame],
    id2label: Dict[int, str],
) -> pd.DataFrame:
    """Compute exact counts and proportions for each intent label per split.

    Args:
        dfs: Mapping of split name to DataFrame.
        id2label: Label ID to human-readable string mapping.

    Returns:
        DataFrame with counts and percentage breakdowns.
    """
    records = []
    for split_name, df in dfs.items():
        total = len(df)
        counts = df["label"].value_counts()
        for label_id, label_name in sorted(id2label.items()):
            cnt = int(counts.get(label_id, 0))
            pct = round((cnt / total) * 100, 2)
            records.append({
                "split": split_name,
                "label_id": label_id,
                "label_name": label_name,
                "count": cnt,
                "percentage": pct,
            })
    return pd.DataFrame(records)


def analyze_sections(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform in-depth analysis of the noisy `sectionName` field.

    Identifies missing values, cardinality, singletons (hapax legomena),
    and common structural noise patterns (numbers, punctuation, casing).

    Args:
        df: Training DataFrame.

    Returns:
        Dictionary containing statistical metrics and top patterns.
    """
    raw_sections = df["sectionName"].fillna("")
    total = len(df)
    missing_count = int((raw_sections == "").sum())

    non_empty = raw_sections[raw_sections != ""]
    unique_count = int(non_empty.nunique())
    val_counts = non_empty.value_counts()
    singletons = int((val_counts == 1).sum())

    # Detect common noise patterns
    has_leading_numbers = int(non_empty.str.contains(r"^\s*\d+[\.\d\s]*", regex=True).sum())
    has_roman_numerals = int(non_empty.str.contains(r"^\s*[IVXLCDM]+[\.\s]", regex=True).sum())
    all_uppercase = int(non_empty.str.isupper().sum())
    all_lowercase = int(non_empty.str.islower().sum())

    return {
        "total_samples": total,
        "missing_or_empty": missing_count,
        "missing_percentage": round((missing_count / total) * 100, 2),
        "unique_raw_sections": unique_count,
        "singletons_appearing_once": singletons,
        "singletons_percentage": round((singletons / unique_count) * 100, 2) if unique_count else 0,
        "noise_leading_numbers": has_leading_numbers,
        "noise_leading_numbers_pct": round((has_leading_numbers / len(non_empty)) * 100, 2),
        "noise_roman_numerals": has_roman_numerals,
        "noise_roman_numerals_pct": round((has_roman_numerals / len(non_empty)) * 100, 2),
        "all_uppercase": all_uppercase,
        "all_uppercase_pct": round((all_uppercase / len(non_empty)) * 100, 2),
        "all_lowercase": all_lowercase,
        "top_20_raw_sections": val_counts.head(20).to_dict(),
    }
