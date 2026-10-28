"""Unified CLI pipeline runner for SciCite Citation Intent Classification."""

import argparse
from pathlib import Path
import pandas as pd

from src.data import get_data_splits
from src.utils import set_seed, plot_confusion_matrices
from src.models import train_baseline, train_transformer


def main():
    parser = argparse.ArgumentParser(description="SciCite Pipeline Runner")
    parser.add_argument("--train-transformers", action="store_true", help="Fine-tune Transformer models (GPU recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    project_root = Path(__file__).resolve().parent

    print("=" * 70)
    print("SciCite Citation Intent Classification - Pipeline Runner")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/3] Loading dataset splits...")
    train_df, val_df, test_df = get_data_splits(project_root)
    print(f"Loaded: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    # 2. Baseline Models
    print("\n[2/3] Evaluating TF-IDF + Logistic Regression Baseline...")
    _, pred_text, m_text = train_baseline(train_df, val_df, test_df, text_column="text_only", seed=args.seed)
    _, pred_sec, m_sec = train_baseline(train_df, val_df, test_df, text_column="text_section_aware", seed=args.seed)

    print(f"  - Baseline (Text-only):      Macro-F1 = {m_text['macro_f1']:.4f} | Accuracy = {m_text['accuracy']:.4f}")
    print(f"  - Baseline (Section-aware):  Macro-F1 = {m_sec['macro_f1']:.4f} | Accuracy = {m_sec['accuracy']:.4f}")

    # 3. Transformer Models (Optional)
    if args.train_transformers:
        print("\n[3/3] Fine-tuning Transformer models (DistilBERT, ModernBERT, SciBERT)...")
        models = [
            ("distilbert", "distilbert-base-uncased"),
            ("modernbert", "answerdotai/ModernBERT-base"),
            ("scibert", "allenai/scibert_scivocab_uncased"),
        ]
        exp_dir = project_root / "experiments"
        for key, hf_name in models:
            for rep in ["text_only", "text_section_aware"]:
                out_dir = exp_dir / f"{key}_{rep}"
                print(f"\nTraining {hf_name} [{rep}]...")
                _, _, t_metrics = train_transformer(
                    model_name=hf_name,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    text_column=rep,
                    output_dir=out_dir,
                    seed=args.seed,
                )
                print(f"  -> Test Macro-F1: {t_metrics['macro_f1']:.4f} | Accuracy: {t_metrics['accuracy']:.4f}")
    else:
        print("\n[3/3] Skipping Transformer training (pre-computed results available in outputs/reports/).")
        print("      To run full training, use flag: --train-transformers")

    print("\n" + "=" * 70)
    print("Execution completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
