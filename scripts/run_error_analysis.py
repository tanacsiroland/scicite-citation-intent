"""Executable CLI script to perform in-depth quantitative and qualitative error analysis."""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Force UTF-8 stdout on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.data_processing.loader import get_label_mappings


def main():
    print("=" * 80)
    print("SciCite Citation Intent Classification - Step 5: In-Depth Error Analysis")
    print("=" * 80)

    set_seed(42)

    errors_dir = PROJECT_ROOT / "outputs" / "errors"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"

    errors_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and merge predictions
    print("\n[1/5] Loading and merging model predictions on official Test set...")
    baseline_pred_path = errors_dir / "baseline_predictions_test.csv"
    transformer_pred_path = errors_dir / "transformer_predictions_test.csv"

    b_df = pd.read_csv(baseline_pred_path)
    t_df = pd.read_csv(transformer_pred_path)

    cols_to_drop = ["id", "label", "label_name", "section_clean", "text_only", "text_section_aware"]
    merged = b_df.merge(t_df.drop(columns=[c for c in cols_to_drop if c in t_df.columns]), on="unique_id")

    id2label, label2id = get_label_mappings()
    total_samples = len(merged)
    print(f"Total merged test instances: {total_samples:,}")

    # 2. Compute error masks for key models
    key_models = {
        "Baseline (Text-only)": "pred_text_only",
        "Baseline (Section-aware)": "pred_text_section_aware",
        "DistilBERT (Section-aware)": "pred_distilbert_text_section_aware",
        "ModernBERT (Section-aware)": "pred_modernbert_text_section_aware",
        "SciBERT (Section-aware)": "pred_scibert_text_section_aware",
    }

    for name, col in key_models.items():
        merged[f"is_error_{col}"] = (merged[col] != merged["label"]).astype(int)

    # 3. Analyze error rates across canonical section categories
    print("\n[2/5] Quantifying error rates across canonical section types...")
    section_records = []
    for sec_name, group in merged.groupby("section_clean"):
        sec_total = len(group)
        rec = {
            "Canonical Section": sec_name,
            "Count": sec_total,
            "Dataset Share (%)": round(sec_total / total_samples * 100, 2),
        }
        for name, col in key_models.items():
            err_cnt = group[f"is_error_{col}"].sum()
            rec[f"Error Rate - {name} (%)"] = round(err_cnt / sec_total * 100, 2)
        section_records.append(rec)

    section_err_df = pd.DataFrame(section_records).sort_values("Count", ascending=False)
    section_err_df.to_csv(reports_dir / "error_rate_by_section.csv", index=False)
    print(section_err_df.to_string(index=False))

    # Plot error rates by section
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = []
    for _, row in section_err_df.iterrows():
        sec = row["Canonical Section"]
        plot_df.append({"Section": sec, "Model": "Baseline (TF-IDF)", "Error Rate (%)": row["Error Rate - Baseline (Section-aware) (%)"]})
        plot_df.append({"Section": sec, "Model": "ModernBERT", "Error Rate (%)": row["Error Rate - ModernBERT (Section-aware) (%)"]})
        plot_df.append({"Section": sec, "Model": "SciBERT", "Error Rate (%)": row["Error Rate - SciBERT (Section-aware) (%)"]})
    plot_df = pd.DataFrame(plot_df)

    sns.barplot(data=plot_df, x="Section", y="Error Rate (%)", hue="Model", palette="muted", ax=ax)
    ax.set_title("Error Rate Across Canonical Section Categories (Test Split)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Error Rate (%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalized Canonical Section", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "error_rate_by_section.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure: {figures_dir / 'error_rate_by_section.png'}")

    # 4. Error transitions breakdown
    print("\n[3/5] Computing error transition confusion matrices...")
    transition_records = []
    for model_name, col in [
        ("Baseline (TF-IDF)", "pred_text_section_aware"),
        ("ModernBERT", "pred_modernbert_text_section_aware"),
        ("SciBERT", "pred_scibert_text_section_aware"),
    ]:
        for true_id, true_name in id2label.items():
            true_mask = (merged["label"] == true_id)
            total_true = true_mask.sum()
            for pred_id, pred_name in id2label.items():
                if true_id != pred_id:
                    cnt = ((merged["label"] == true_id) & (merged[col] == pred_id)).sum()
                    pct = round(cnt / total_true * 100, 2)
                    transition_records.append({
                        "Model": model_name,
                        "True Class": true_name,
                        "Predicted Class": pred_name,
                        "Error Count": cnt,
                        "Error Rate within Class (%)": pct,
                    })

    trans_df = pd.DataFrame(transition_records)
    trans_df.to_csv(reports_dir / "error_transitions_breakdown.csv", index=False)
    print(trans_df.to_string(index=False))

    # 5. Extract categorized error subsets
    print("\n[4/5] Isolating representative error subsets for qualitative analysis...")

    # A) Baseline errors rescued by ModernBERT / SciBERT
    base_wrong = (merged["pred_text_section_aware"] != merged["label"])
    trans_right = (merged["pred_modernbert_text_section_aware"] == merged["label"]) & (merged["pred_scibert_text_section_aware"] == merged["label"])
    rescued_df = merged[base_wrong & trans_right].copy()
    rescued_df.to_csv(errors_dir / "baseline_errors_fixed_by_transformers.csv", index=False)
    print(f"Baseline errors successfully fixed by Transformers: {len(rescued_df):,} instances")

    # B) Universal errors (all models failed)
    all_wrong_mask = (
        (merged["pred_text_section_aware"] != merged["label"]) &
        (merged["pred_distilbert_text_section_aware"] != merged["label"]) &
        (merged["pred_modernbert_text_section_aware"] != merged["label"]) &
        (merged["pred_scibert_text_section_aware"] != merged["label"])
    )
    hard_df = merged[all_wrong_mask].copy()
    hard_df.to_csv(errors_dir / "hard_instances_all_models_wrong.csv", index=False)
    print(f"Universal 'hard' instances (all models failed): {len(hard_df):,} instances ({len(hard_df)/total_samples*100:.2f}%)")

    # 6. Build Curated Qualitative Case Studies Table (Ready for documentation inclusion)
    print("\n[5/5] Building curated qualitative case studies for documentation...")
    case_studies = []

    # Category 1: Context-rescued (section context disambiguates methodological sentence)
    sample_c1 = rescued_df[rescued_df["label_name"] == "method"].head(2)
    for _, r in sample_c1.iterrows():
        case_studies.append({
            "Error Category": "1. Context-Rescued (Method)",
            "True Label": r["label_name"],
            "Section Header": r["section_clean"],
            "Baseline Pred": r["pred_label_text_section_aware"],
            "ModernBERT Pred": r["pred_label_modernbert_text_section_aware"],
            "SciBERT Pred": r["pred_label_scibert_text_section_aware"],
            "Citation Context": r["text_only"],
            "Linguistic Explanation": "Sentence describes background terminology but appears in Methods section; section context correctly steers the transformer.",
        })

    # Category 2: Result vs Background Boundary Ambiguity
    sample_c2 = hard_df[(hard_df["label_name"] == "result") & (hard_df["section_clean"] == "discussion")].head(2)
    for _, r in sample_c2.iterrows():
        case_studies.append({
            "Error Category": "2. Result vs. Background Boundary",
            "True Label": r["label_name"],
            "Section Header": r["section_clean"],
            "Baseline Pred": r["pred_label_text_section_aware"],
            "ModernBERT Pred": r["pred_label_modernbert_text_section_aware"],
            "SciBERT Pred": r["pred_label_scibert_text_section_aware"],
            "Citation Context": r["text_only"],
            "Linguistic Explanation": "Citing past literature to contrast with own findings creates dual intent; models default to background due to prior citations.",
        })

    # Category 3: Missing Context Penalty
    sample_c3 = hard_df[hard_df["section_clean"] == "missing"].head(2)
    for _, r in sample_c3.iterrows():
        case_studies.append({
            "Error Category": "3. Missing Section Penalty",
            "True Label": r["label_name"],
            "Section Header": r["section_clean"],
            "Baseline Pred": r["pred_label_text_section_aware"],
            "ModernBERT Pred": r["pred_label_modernbert_text_section_aware"],
            "SciBERT Pred": r["pred_label_scibert_text_section_aware"],
            "Citation Context": r["text_only"],
            "Linguistic Explanation": "Without section header, model lacks structural prior and fails on lexically sparse sentence.",
        })

    case_df = pd.DataFrame(case_studies)
    case_df.to_csv(errors_dir / "qualitative_case_studies.csv", index=False)
    print(f"Saved {len(case_df)} curated case studies to {errors_dir / 'qualitative_case_studies.csv'}")

    print("\n" + "=" * 80)
    print("Error Analysis completed successfully! All reports and error subsets exported.")
    print("=" * 80)


if __name__ == "__main__":
    main()
