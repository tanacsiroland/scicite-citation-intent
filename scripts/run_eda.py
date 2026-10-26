"""Executable CLI script to run Exploratory Data Analysis (EDA) on SciCite."""

import os
import sys
import json
from pathlib import Path

# Force UTF-8 stdout on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.data_processing.loader import load_scicite_dataset, get_label_mappings, dataset_to_dataframes
from src.data_processing.eda import (
    compute_split_summary,
    compute_class_distribution,
    analyze_sections,
)
from src.utils.visualizer import (
    plot_class_distribution,
    plot_top_sections,
    plot_section_label_contingency,
)


def main():
    print("=" * 70)
    print("SciCite Citation Intent Classification - Step 1: Exploratory Data Analysis")
    print("=" * 70)

    # 1. Reproducibility
    set_seed(42)

    # 2. Paths setup
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load dataset
    print("\n[1/4] Fetching official 'allenai/scicite' dataset from Hugging Face...")
    dataset = load_scicite_dataset()
    id2label, label2id = get_label_mappings(dataset)
    print(f"Loaded splits: {list(dataset.keys())}")
    print(f"Label mapping: {id2label}")

    dfs = dataset_to_dataframes(dataset)

    # 4. Split summary
    print("\n[2/4] Computing dataset split summary and missingness...")
    split_summary = compute_split_summary(dfs)
    print(split_summary.to_string(index=False))
    split_summary.to_csv(reports_dir / "eda_split_summary.csv", index=False)

    # 5. Class distribution
    print("\n[3/4] Computing class distributions...")
    class_dist = compute_class_distribution(dfs, id2label)
    print(class_dist.to_string(index=False))
    class_dist.to_csv(reports_dir / "eda_class_distribution.csv", index=False)

    plot_class_distribution(dfs, id2label, output_path=str(figures_dir / "class_distribution.png"))
    print(f"Saved figure: {figures_dir / 'class_distribution.png'}")

    # 4. Section analysis
    print("\n[4/4] In-depth analysis of raw 'sectionName' field...")
    section_analysis = analyze_sections(dfs["train"])

    with open(reports_dir / "eda_section_analysis.json", "w", encoding="utf-8") as f:
        json.dump(section_analysis, f, indent=2)

    print(f"Total training samples: {section_analysis['total_samples']}")
    print(f"Missing / Empty sectionName: {section_analysis['missing_or_empty']} ({section_analysis['missing_percentage']}%)")
    print(f"Unique raw section headers: {section_analysis['unique_raw_sections']}")
    print(f"Singletons (appear only 1x): {section_analysis['singletons_appearing_once']} ({section_analysis['singletons_percentage']}% of unique headers)")
    print(f"Headers with numeric prefixes: {section_analysis['noise_leading_numbers']} ({section_analysis['noise_leading_numbers_pct']}%)")
    print(f"Headers in all uppercase: {section_analysis['all_uppercase']} ({section_analysis['all_uppercase_pct']}%)")

    plot_top_sections(dfs["train"], top_n=15, output_path=str(figures_dir / "top_raw_sections.png"))
    print(f"Saved figure: {figures_dir / 'top_raw_sections.png'}")

    plot_section_label_contingency(dfs["train"], id2label, top_n=10, output_path=str(figures_dir / "section_intent_contingency.png"))
    print(f"Saved figure: {figures_dir / 'section_intent_contingency.png'}")

    print("\n" + "=" * 70)
    print("EDA Step Completed successfully! All artifacts exported to outputs/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
