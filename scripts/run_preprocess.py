"""Executable CLI script to run preprocessing and generate comparative representations."""

import sys
from pathlib import Path
import pandas as pd

# Force UTF-8 stdout on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.data_processing.loader import load_scicite_dataset, dataset_to_dataframes
from src.data_processing.preprocess import (
    build_input_representations,
    preprocess_dataset,
    CANONICAL_SECTIONS,
)
from src.utils.visualizer import (
    plot_canonical_section_distribution,
    plot_canonical_section_contingency,
)


def main():
    print("=" * 75)
    print("SciCite Citation Intent Classification - Step 2: Data Preprocessing Pipeline")
    print("=" * 75)

    # 1. Reproducibility
    set_seed(42)

    # 2. Directory setup
    processed_data_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    processed_data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load raw datasets
    print("\n[1/4] Loading raw SciCite datasets...")
    dataset = load_scicite_dataset()
    raw_dfs = dataset_to_dataframes(dataset)

    # 4. Process and build comparative representations
    print("\n[2/4] Normalizing section headers & generating comparative representations...")
    processed_dfs = preprocess_dataset(raw_dfs=raw_dfs, output_dir=processed_data_dir)

    # 5. Export summary reports
    print("\n[3/4] Exporting statistical distributions and contingency tables...")
    records = []
    for split_name, df in processed_dfs.items():
        counts = df["section_clean"].value_counts()
        total = len(df)
        for sec in CANONICAL_SECTIONS:
            cnt = int(counts.get(sec, 0))
            records.append({
                "split": split_name,
                "canonical_section": sec,
                "count": cnt,
                "percentage": round(cnt / total * 100, 2),
            })

    dist_df = pd.DataFrame(records)
    dist_df.to_csv(reports_dir / "canonical_sections_distribution.csv", index=False)
    print(f"Saved distribution report: {reports_dir / 'canonical_sections_distribution.csv'}")

    contingency = (
        pd.crosstab(
            processed_dfs["train"]["section_clean"],
            processed_dfs["train"]["label_name"],
            normalize="index",
        )
        * 100
    ).round(2)
    contingency.to_csv(reports_dir / "canonical_section_contingency.csv")
    print(f"Saved contingency report: {reports_dir / 'canonical_section_contingency.csv'}")
    print("\nContingency P(Intent | Normalized Section) [%]:")
    print(contingency.to_string())

    # 6. Render figures
    print("\n[4/4] Generating publication-grade figures...")
    plot_canonical_section_distribution(
        processed_dfs,
        output_path=str(figures_dir / "canonical_sections_distribution.png"),
    )
    print(f"Saved figure: {figures_dir / 'canonical_sections_distribution.png'}")

    plot_canonical_section_contingency(
        processed_dfs["train"],
        output_path=str(figures_dir / "canonical_section_contingency.png"),
    )
    print(f"Saved figure: {figures_dir / 'canonical_section_contingency.png'}")

    # Inspect sample pairs
    print("\n--- Example Comparative Representations (First 2 Train Instances) ---")
    for idx, row in processed_dfs["train"].head(2).iterrows():
        print(f"\nSample #{idx + 1} (True Intent: {row['label_name']}):")
        print(f"  Raw sectionName:       '{row['sectionName']}'")
        print(f"  Normalized section:    '{row['section_clean']}'")
        print(f"  [Text-only]:           {row['text_only'][:100]}...")
        print(f"  [Section-aware]:       {row['text_section_aware'][:120]}...")

    print("\n" + "=" * 75)
    print("Preprocessing completed successfully! Processed splits stored in data/processed/.")
    print("=" * 75)


if __name__ == "__main__":
    main()
