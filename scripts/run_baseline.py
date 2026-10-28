"""Executable CLI script to train and evaluate the TF-IDF Baseline model."""

import sys
import json
from pathlib import Path
import pandas as pd
import yaml

# Force UTF-8 stdout on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.data_processing.loader import get_label_mappings
from src.models.baseline import TfidfBaseline, tune_baseline_hyperparameters
from src.utils.metrics import metrics_to_dataframe
from src.utils.visualizer import plot_confusion_matrix


def main():
    print("=" * 75)
    print("SciCite Citation Intent Classification - Step 3: TF-IDF Baseline Model")
    print("=" * 75)

    # 1. Reproducibility & Config
    config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    # 2. Directory setup
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    errors_dir = PROJECT_ROOT / "outputs" / "errors"

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load preprocessed splits
    print("\n[1/5] Loading preprocessed datasets from data/processed/...")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "validation.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    id2label, label2id = get_label_mappings()

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,} | Test samples: {len(test_df):,}")

    tfidf_params = config.get("data", {}).get("tfidf", {})
    representations = ["text_only", "text_section_aware"]

    comparison_records = []
    tuning_records = []
    trained_models = {}
    test_eval_metrics = {}

    test_predictions_df = test_df[["id", "unique_id", "label", "label_name", "section_clean", "text_only", "text_section_aware"]].copy()

    # 4. Hyperparameter tuning & model evaluation
    for rep_name in representations:
        rep_label = "Text-only" if rep_name == "text_only" else "Section-aware"
        print(f"\n" + "-" * 60)
        print(f"Evaluating Representation: [{rep_label}] (Column: {rep_name})")
        print("-" * 60)

        # Hyperparameter search on validation set
        print(f"[2/5] Tuning regularization C & class_weight on Validation set...")
        best_params, tune_df = tune_baseline_hyperparameters(
            X_train=train_df[rep_name],
            y_train=train_df["label"],
            X_val=val_df[rep_name],
            y_val=val_df["label"],
            id2label=id2label,
            c_grid=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            class_weights=["balanced", None],
            tfidf_params=tfidf_params,
            random_state=seed,
        )
        tune_df["representation"] = rep_name
        tuning_records.append(tune_df)

        print(f"Optimal parameters for {rep_label}: C={best_params['C']}, class_weight='{best_params['class_weight']}'")
        print(f"Top 3 validation configurations:\n{tune_df.head(3).to_string(index=False)}")

        # Train model with best parameters
        print(f"\n[3/5] Fitting optimal {rep_label} model on full training set...")
        optimal_lr_params = {
            "C": best_params["C"],
            "class_weight": best_params["class_weight"],
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": seed,
        }
        model = TfidfBaseline(tfidf_params=tfidf_params, lr_params=optimal_lr_params, random_state=seed)
        model.fit(train_df[rep_name], train_df["label"])
        trained_models[rep_name] = model

        # Evaluate on Validation set
        val_metrics = model.evaluate(val_df[rep_name], val_df["label"], id2label=id2label)
        # Evaluate on Test set
        test_metrics = model.evaluate(test_df[rep_name], test_df["label"], id2label=id2label)
        test_eval_metrics[rep_name] = test_metrics

        # Record predictions
        test_preds = model.predict(test_df[rep_name])
        test_predictions_df[f"pred_{rep_name}"] = test_preds
        test_predictions_df[f"pred_label_{rep_name}"] = [id2label[p] for p in test_preds]

        # Save individual metrics JSON
        metrics_filename = f"baseline_{rep_name}_metrics.json"
        with open(reports_dir / metrics_filename, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        # Save tabular classification report
        report_df = metrics_to_dataframe(test_metrics)
        report_path = reports_dir / f"baseline_{rep_name}_classification_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"Classification Report ({rep_label}):\n{report_df.to_string(index=False)}")

        # Plot confusion matrix
        cm_fig_path = figures_dir / f"baseline_confusion_matrix_{rep_name}.png"
        plot_confusion_matrix(
            cm=test_metrics["confusion_matrix"],
            target_names=test_metrics["target_names"],
            title=f"Baseline TF-IDF + LR Confusion Matrix ({rep_label})",
            output_path=str(cm_fig_path),
        )
        print(f"Saved confusion matrix: {cm_fig_path}")

        # Summary entry
        comparison_records.append({
            "Representation": rep_label,
            "Accuracy": test_metrics["accuracy"],
            "Macro P": test_metrics["macro_precision"],
            "Macro R": test_metrics["macro_recall"],
            "Macro F1": test_metrics["macro_f1"],
            "Micro P": test_metrics["micro_precision"],
            "Micro R": test_metrics["micro_recall"],
            "Micro F1": test_metrics["micro_f1"],
            "Weighted P": test_metrics["weighted_precision"],
            "Weighted R": test_metrics["weighted_recall"],
            "Weighted F1": test_metrics["weighted_f1"],
            "Method F1": test_metrics["per_class"][0]["f1"],
            "Background F1": test_metrics["per_class"][1]["f1"],
            "Result F1": test_metrics["per_class"][2]["f1"],
            "Best C": best_params["C"],
            "Class Weight": "balanced" if best_params["class_weight"] == "balanced" else "none",
        })

    # 5. Export comprehensive summary
    print("\n" + "=" * 75)
    print("[4/5] Comparative Evaluation Summary Across Representations")
    print("=" * 75)
    summary_df = pd.DataFrame(comparison_records)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(reports_dir / "baseline_comparison_summary.csv", index=False)

    all_tuning_df = pd.concat(tuning_records, ignore_index=True)
    all_tuning_df.to_csv(reports_dir / "baseline_hyperparameter_tuning.csv", index=False)

    # 6. Save test predictions for error analysis
    print("\n[5/5] Exporting test predictions for subsequent error analysis...")
    test_predictions_df.to_csv(errors_dir / "baseline_predictions_test.csv", index=False)
    print(f"Saved predictions to {errors_dir / 'baseline_predictions_test.csv'}")

    print("\n" + "=" * 75)
    print("Baseline phase completed successfully! All reports and figures exported.")
    print("=" * 75)


if __name__ == "__main__":
    main()
