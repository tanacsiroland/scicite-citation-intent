"""Data loading, cleaning, and representation pipeline for SciCite."""

import os
import re
import json
import urllib.request
import tarfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd

DATASET_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
CLASS_NAMES = ["method", "background", "result"]
LABEL2ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID2LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}

CANONICAL_SECTIONS = [
    "introduction",
    "method",
    "result",
    "discussion",
    "result_discussion",
    "other",
    "missing",
]

_SECTION_KEYWORDS = {
    "introduction": [
        "intro", "background", "related work", "prior work", "overview", "motivation", "literature", "preliminar"
    ],
    "method": [
        "method", "material", "procedure", "approach", "implementation", "model", "algorithm",
        "technique", "experimental setup", "experimental design", "data collection", "formulation"
    ],
    "result": [
        "result", "finding", "experiment", "evaluation", "empirical", "performance", "benchmark", "observation"
    ],
    "discussion": [
        "discussion", "conclusion", "concluding", "limitation", "future work", "summary", "closing"
    ],
}


def normalize_section_name(raw_name: Optional[str]) -> str:
    """Normalize noisy raw section headers into canonical academic categories."""
    if raw_name is None:
        return "missing"
    s = str(raw_name).strip().lower()
    if not s or s in ("nan", "none", "null", "unknown"):
        return "missing"

    s = re.sub(r"^[0-9ivxlcdm]+[\.\s\:\-\)]+", "", s).strip()
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s).strip()
    if not s:
        return "missing"

    if "result" in s and "discussion" in s:
        return "result_discussion"

    for category, keywords in _SECTION_KEYWORDS.items():
        if any(kw in s for kw in keywords):
            return category

    return "other"


def build_representations(df: pd.DataFrame) -> pd.DataFrame:
    """Build comparative text-only and section-aware input representations."""
    processed = df.copy()
    if "section_clean" not in processed.columns:
        processed["section_clean"] = processed["sectionName"].apply(normalize_section_name)

    processed["text_only"] = processed["string"].fillna("").astype(str).str.strip()
    processed["text_section_aware"] = (
        "Section: " + processed["section_clean"] + ". Citation: " + processed["text_only"]
    )
    return processed


def ensure_raw_scicite_data(raw_dir: Path) -> Dict[str, Path]:
    """Ensure official SciCite train/dev/test jsonl files are present locally."""
    scicite_dir = raw_dir / "scicite"
    splits = {
        "train": scicite_dir / "train.jsonl",
        "validation": scicite_dir / "dev.jsonl",
        "test": scicite_dir / "test.jsonl",
    }
    if not all(p.is_file() for p in splits.values()):
        raw_dir.mkdir(parents=True, exist_ok=True)
        tar_path = raw_dir / "scicite.tar.gz"
        if not tar_path.is_file():
            print(f"Downloading SciCite archive from {DATASET_URL}...")
            urllib.request.urlretrieve(DATASET_URL, tar_path)
        print("Extracting SciCite train/dev/test jsonl...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name in ("scicite/train.jsonl", "scicite/dev.jsonl", "scicite/test.jsonl"):
                    tar.extract(member, path=raw_dir)
    return splits


def load_split_dataframe(filepath: Path) -> pd.DataFrame:
    """Read a SciCite JSONL file and deduplicate records by unique_id."""
    rows = []
    seen = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            uid = str(d.get("unique_id", d.get("id")))
            if uid in seen:
                continue
            seen.add(uid)
            label_str = str(d.get("label", "")).strip().lower()
            if label_str not in LABEL2ID:
                continue
            rows.append({
                "string": str(d.get("string", "")),
                "sectionName": str(d.get("sectionName", "") or ""),
                "label": LABEL2ID[label_str],
                "label_name": label_str,
                "citingPaperId": str(d.get("citingPaperId", "")),
                "citedPaperId": str(d.get("citedPaperId", "")),
                "id": str(d.get("id", "")),
                "unique_id": uid,
            })
    return pd.DataFrame(rows)


def get_data_splits(project_root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train, validation, and test splits (cached or processed on the fly)."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    proc_dir = project_root / "data" / "processed"
    train_pq = proc_dir / "train.parquet"
    val_pq = proc_dir / "validation.parquet"
    test_pq = proc_dir / "test.parquet"

    if train_pq.is_file() and val_pq.is_file() and test_pq.is_file():
        return pd.read_parquet(train_pq), pd.read_parquet(val_pq), pd.read_parquet(test_pq)

    raw_dir = project_root / "data" / "raw"
    split_paths = ensure_raw_scicite_data(raw_dir)
    train_df = build_representations(load_split_dataframe(split_paths["train"]))
    val_df = build_representations(load_split_dataframe(split_paths["validation"]))
    test_df = build_representations(load_split_dataframe(split_paths["test"]))

    proc_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_pq, index=False)
    val_df.to_parquet(val_pq, index=False)
    test_df.to_parquet(test_pq, index=False)

    return train_df, val_df, test_df
