"""Data preprocessing and representation pipeline for SciCite."""

import re
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
from datasets import Dataset, DatasetDict

from src.data_processing.loader import load_scicite_dataset, dataset_to_dataframes


# Canonical section categories
CANONICAL_SECTIONS = [
    "introduction",
    "method",
    "result",
    "discussion",
    "result_discussion",
    "other",
    "missing",
]

# Section keyword mapping definitions
_SECTION_KEYWORDS = {
    "introduction": [
        "intro",
        "background",
        "related work",
        "prior work",
        "overview",
        "motivation",
        "literature",
        "preliminar",
    ],
    "method": [
        "method",
        "material",
        "procedure",
        "approach",
        "implementation",
        "model",
        "algorithm",
        "technique",
        "experimental setup",
        "experimental design",
        "data collection",
        "formulation",
    ],
    "result": [
        "result",
        "finding",
        "experiment",
        "evaluation",
        "empirical",
        "performance",
        "benchmark",
        "observation",
    ],
    "discussion": [
        "discussion",
        "conclusion",
        "concluding",
        "limitation",
        "future work",
        "summary",
        "closing",
    ],
}


def normalize_section_name(raw_name: Optional[str]) -> str:
    """Normalize noisy raw section headers into canonical academic categories.

    Pipeline:
      1. Handle null, empty, or placeholder strings -> 'missing'.
      2. Strip leading numeric prefixes (e.g., '1. ', '2.1. ', 'IV. ').
      3. Strip non-alphanumeric punctuation boundaries.
      4. Detect composite 'result and discussion' sections.
      5. Map to core academic sections: introduction, method, result, discussion.
      6. Fallback unmapped titles -> 'other'.

    Args:
        raw_name: Raw section header string from scientific publication.

    Returns:
        One of the canonical section names:
        ('introduction', 'method', 'result', 'discussion', 'result_discussion', 'other', 'missing')
    """
    if raw_name is None:
        return "missing"

    s = str(raw_name).strip().lower()
    if not s or s in ("nan", "none", "null", "unknown"):
        return "missing"

    # Remove leading numbering schemes (e.g. '1. ', '1.2.3 ', 'IV. ', 'B. ')
    s = re.sub(r"^[0-9ivxlcdm]+[\.\s\:\-\)]+", "", s).strip()
    # Strip leading/trailing punctuation
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s).strip()

    if not s:
        return "missing"

    # Check for composite Result and Discussion sections
    if "result" in s and "discussion" in s:
        return "result_discussion"

    # Match canonical keywords in priority order
    for category, keywords in _SECTION_KEYWORDS.items():
        if any(kw in s for kw in keywords):
            return category

    return "other"


def build_input_representations(df: pd.DataFrame) -> pd.DataFrame:
    """Construct comparative input text representations for baseline and neural models.

    Generates:
      - text_only: The raw citation context sentence.
      - text_section_aware: Standardized prefix conditioning format:
        "Section: {canonical_section}. Citation: {citation_sentence}"

    Args:
        df: DataFrame containing at least 'string' and 'sectionName' columns.

    Returns:
        DataFrame augmented with 'section_clean', 'text_only', and 'text_section_aware'.
    """
    processed = df.copy()

    # Normalize section headers
    processed["section_clean"] = processed["sectionName"].apply(normalize_section_name)

    # 1. Text-only representation (raw citation sentence)
    processed["text_only"] = processed["string"].fillna("").astype(str).str.strip()

    # 2. Section-aware representation (context-enhanced prefix conditioning)
    processed["text_section_aware"] = (
        "Section: "
        + processed["section_clean"]
        + ". Citation: "
        + processed["text_only"]
    )

    return processed


def preprocess_dataset(
    raw_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Run full preprocessing pipeline across all splits and persist processed artifacts.

    Args:
        raw_dfs: Optional mapping of split name to raw DataFrame. If None, loads via loader.
        output_dir: Directory where processed parquets are saved (defaults to data/processed).

    Returns:
        Dictionary of preprocessed DataFrames per split.
    """
    if raw_dfs is None:
        dataset = load_scicite_dataset()
        raw_dfs = dataset_to_dataframes(dataset)

    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dfs = {}

    for split_name, df in raw_dfs.items():
        processed_df = build_input_representations(df)
        parquet_path = output_dir / f"{split_name}.parquet"
        csv_path = output_dir / f"{split_name}.csv"

        processed_df.to_parquet(parquet_path, index=False)
        processed_df.to_csv(csv_path, index=False)

        processed_dfs[split_name] = processed_df
        print(f"Saved {split_name} split ({len(processed_df)} records) to {parquet_path}")

    return processed_dfs
