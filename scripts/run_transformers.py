"""Executable CLI script to fine-tune and evaluate Transformer models (DistilBERT, ModernBERT, SciBERT)."""

import os
import sys
import json
import time
from pathlib import Path
import yaml
import pandas as pd
import torch

# Force UTF-8 stdout on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.data_processing.loader import get_label_mappings
from src.models.transformer import train_and_evaluate_transformer
from src.utils.metrics import metrics_to_dataframe
from src.utils.visualizer import plot_confusion_matrix, plot_training_curves


def main():
    print("=" * 80)
    print("SciCite Citation Intent Classification - Step 4: Transformer Fine-Tuning")
    print("=" * 80)

    # 1. Configuration & Reproducibility
    config_path = PROJECT_ROOT / "configs" / "transformer.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 2. Directory setup
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    errors_dir = PROJECT_ROOT / "outputs" / "errors"
    experiments_dir = PROJECT_ROOT / "experiments"

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load preprocessed splits
    print("\nLoading preprocessed datasets...")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "validation.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    id2label, label2id = get_label_mappings()

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    models_cfg = config.get("models", {})
    training_base_cfg = config.get("training", {})
    representations = config.get("representations", {}).get("modes", ["text_only", "text_section_aware"])

    results_table = []
    all_histories = {}
    predictions_df = test_df[["id", "unique_id", "label", "label_name", "section_clean", "text_only", "text_section_aware"]].copy()

    total_runs = len(models_cfg) * len(representations)
    run_idx = 1

    # 4. Training loop over Model x Representation matrix
    for model_key, model_info in models_cfg.items():
        model_name = model_info["name"]
        display_name = model_info.get("display_name", model_key.upper())
        model_lr = model_info.get("learning_rate", training_base_cfg.get("learning_rate", 2e-5))

        for rep_col in representations:
            rep_label = "Text-only" if rep_col == "text_only" else "Section-aware"
            run_name = f"{model_key}_{rep_col}"

            print("\n" + "=" * 80)
            print(f"[{run_idx}/{total_runs}] Model: {display_name} | Representation: {rep_label} ({rep_col})")
            print(f"Learning Rate: {model_lr} | Max Length: {training_base_cfg.get('max_length', 512)} | Batch Size: {training_base_cfg.get('per_device_train_batch_size', 32)}")
            print("=" * 80)

            run_output_dir = experiments_dir / run_name

            run_training_args = dict(training_base_cfg)
            run_training_args["learning_rate"] = model_lr

            start_time = time.time()
            result = train_and_evaluate_transformer(
                model_name_or_path=model_name,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                text_column=rep_col,
                output_dir=run_output_dir,
                id2label=id2label,
                label2id=label2id,
                training_args_kwargs=run_training_args,
                seed=seed,
            )
            elapsed_time = time.time() - start_time

            metrics = result["test_metrics"]
            preds = result["predictions"]
            probs = result["probabilities"]

            # Store predictions
            predictions_df[f"pred_{run_name}"] = preds
            predictions_df[f"pred_label_{run_name}"] = [id2label[p] for p in preds]

            # Save individual metrics JSON
            metrics_path = reports_dir / f"transformer_{run_name}_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({**metrics, "training_time_sec": round(elapsed_time, 2)}, f, indent=2)

            # Save training log history JSON
            all_histories[run_name] = result["log_history"]
            with open(reports_dir / f"transformer_{run_name}_history.json", "w", encoding="utf-8") as f:
                json.dump(result["log_history"], f, indent=2)

            # Save detailed classification report CSV
            report_df = metrics_to_dataframe(metrics)
            report_path = reports_dir / f"transformer_{run_name}_classification_report.csv"
            report_df.to_csv(report_path, index=False)

            # Plot confusion matrix
            cm_fig_path = figures_dir / f"transformer_confusion_matrix_{run_name}.png"
            plot_confusion_matrix(
                cm=metrics["confusion_matrix"],
                target_names=metrics["target_names"],
                title=f"{display_name} Confusion Matrix ({rep_label})",
                output_path=str(cm_fig_path),
            )
            print(f"Saved confusion matrix: {cm_fig_path}")

            # Append to master summary
            results_table.append({
                "Model": display_name,
                "Representation": rep_label,
                "Accuracy": metrics["accuracy"],
                "Macro P": metrics["macro_precision"],
                "Macro R": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
                "Micro P": metrics["micro_precision"],
                "Micro R": metrics["micro_recall"],
                "Micro F1": metrics["micro_f1"],
                "Weighted P": metrics["weighted_precision"],
                "Weighted R": metrics["weighted_recall"],
                "Weighted F1": metrics["weighted_f1"],
                "Method F1": metrics["per_class"][0]["f1"],
                "Background F1": metrics["per_class"][1]["f1"],
                "Result F1": metrics["per_class"][2]["f1"],
                "Training Time (s)": round(elapsed_time, 1),
            })

            run_idx += 1

    # 5. Export master summary, training curves & predictions
    print("\n" + "=" * 80)
    print("Transformer Comparative Evaluation Leaderboard (10 Epochs)")
    print("=" * 80)

    # Plot multi-epoch learning curves
    curves_fig_path = figures_dir / "transformer_training_curves.png"
    plot_training_curves(all_histories, output_path=str(curves_fig_path))
    print(f"Saved multi-epoch training curves: {curves_fig_path}")

    summary_df = pd.DataFrame(results_table).sort_values("Macro F1", ascending=False)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(reports_dir / "transformers_comparison_summary.csv", index=False)

    predictions_df.to_csv(errors_dir / "transformer_predictions_test.csv", index=False)
    print(f"\nSaved test predictions to: {errors_dir / 'transformer_predictions_test.csv'}")
    print(f"Saved leaderboard summary to: {reports_dir / 'transformers_comparison_summary.csv'}")

    # Build Master Benchmark comparing Baseline + all Transformers
    baseline_summary_path = reports_dir / "baseline_comparison_summary.csv"
    if baseline_summary_path.is_file():
        baseline_df = pd.read_csv(baseline_summary_path)
        baseline_df["Model"] = "Baseline (TF-IDF + LR)"
        # Align columns
        common_cols = [c for c in summary_df.columns if c in baseline_df.columns]
        master_leaderboard = pd.concat([baseline_df[common_cols], summary_df[common_cols]], ignore_index=True)
        master_leaderboard = master_leaderboard.sort_values("Macro F1", ascending=False)
        master_leaderboard.to_csv(reports_dir / "all_models_comparative_summary.csv", index=False)
        print("\n=== Master Comparative Leaderboard (Baseline vs. Transformers) ===")
        print(master_leaderboard.to_string(index=False))

    print("\n" + "=" * 80)
    print("Phase 4 Transformer Fine-Tuning Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
