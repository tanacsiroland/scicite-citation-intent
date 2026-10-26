"""Dataset loader for the official SciCite citation intent corpus."""

import os
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel


DATASET_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
CLASS_NAMES = ["method", "background", "result"]
LABEL2ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID2LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def _ensure_data_files(data_dir: Path) -> Dict[str, Path]:
    """Ensure SciCite jsonl files are downloaded and extracted locally.

    Args:
        data_dir: Target directory path to store extracted dataset files.

    Returns:
        Dictionary mapping split names to file paths.
    """
    scicite_dir = data_dir / "scicite"
    splits = {
        "train": scicite_dir / "train.jsonl",
        "validation": scicite_dir / "dev.jsonl",
        "test": scicite_dir / "test.jsonl",
    }

    all_exist = all(p.is_file() for p in splits.values())
    if not all_exist:
        data_dir.mkdir(parents=True, exist_ok=True)
        tar_path = data_dir / "scicite.tar.gz"

        if not tar_path.is_file():
            print(f"Downloading SciCite archive from {DATASET_URL}...")
            urllib.request.urlretrieve(DATASET_URL, tar_path)

        print(f"Extracting SciCite archive to {data_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)

    return splits


def _read_split_jsonl(filepath: Path) -> pd.DataFrame:
    """Read a SciCite JSONL file and deduplicate records by unique_id.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        Cleaned pandas DataFrame for the split.
    """
    rows = []
    seen_ids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            uid = str(d.get("unique_id", d.get("id")))
            if uid in seen_ids:
                continue
            seen_ids.add(uid)

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
                "excerpt_index": int(d.get("excerpt_index", -1)),
                "isKeyCitation": bool(d.get("isKeyCitation", False)),
                "id": str(d.get("id", "")),
                "unique_id": uid,
            })

    return pd.DataFrame(rows)


def load_scicite_dataset(data_dir: Optional[str] = None) -> DatasetDict:
    """Load the official SciCite dataset into a Hugging Face DatasetDict.

    The official split consists of:
      - train: 8,243 samples
      - validation: 911 samples
      - test: 1,861 samples

    Args:
        data_dir: Optional directory where scicite data is cached. Defaults to `data/raw`.

    Returns:
        DatasetDict with standard 'train', 'validation', and 'test' splits.
    """
    if data_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        raw_dir = project_root / "data" / "raw"
    else:
        raw_dir = Path(data_dir)

    split_paths = _ensure_data_files(raw_dir)

    features = Features({
        "string": Value("string"),
        "sectionName": Value("string"),
        "label": ClassLabel(names=CLASS_NAMES),
        "label_name": Value("string"),
        "citingPaperId": Value("string"),
        "citedPaperId": Value("string"),
        "excerpt_index": Value("int32"),
        "isKeyCitation": Value("bool"),
        "id": Value("string"),
        "unique_id": Value("string"),
    })

    datasets_dict = {}
    for split_name, split_path in split_paths.items():
        df = _read_split_jsonl(split_path)
        ds = Dataset.from_pandas(df, features=features)
        datasets_dict[split_name] = ds

    return DatasetDict(datasets_dict)


def get_label_mappings(dataset: Optional[DatasetDict] = None) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Return ID-to-label and label-to-ID mappings.

    Returns:
        Tuple of (ID2LABEL, LABEL2ID).
    """
    return ID2LABEL.copy(), LABEL2ID.copy()


def dataset_to_dataframes(dataset: DatasetDict) -> Dict[str, pd.DataFrame]:
    """Convert Hugging Face DatasetDict splits into pandas DataFrames.

    Args:
        dataset: Loaded Hugging Face DatasetDict.

    Returns:
        Dictionary mapping split name to pandas DataFrame.
    """
    return {split: dataset[split].to_pandas() for split in dataset.keys()}
