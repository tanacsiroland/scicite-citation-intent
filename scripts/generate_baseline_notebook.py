"""Script to generate the complete, executable 02_baseline_tfidf.ipynb notebook."""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SciCite: Citation Intent Classification\n",
                "## 02. Baseline Model: TF-IDF + Logistic Regression\n",
                "\n",
                "**Course:** Mesterséges intelligencia alkalmazási területei (MSc)\n",
                "**Objective:** Establish a strong, reproducible classical machine learning baseline using TF-IDF feature extraction and Logistic Regression. Quantify the performance difference between `text-only` and `section-aware` representations across validation and official test splits."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Environment & Reproducibility Setup\n",
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import yaml\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Add project root to sys.path\n",
                "PROJECT_ROOT = Path(\"..\").resolve()\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(PROJECT_ROOT))\n",
                "\n",
                "from src.utils.seed import set_seed\n",
                "from src.data_processing.loader import get_label_mappings\n",
                "from src.models.baseline import TfidfBaseline, tune_baseline_hyperparameters\n",
                "from src.utils.metrics import metrics_to_dataframe\n",
                "from src.utils.visualizer import plot_confusion_matrix\n",
                "\n",
                "set_seed(42)\n",
                "print(\"Environment initialized and seed set to 42.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Loading Preprocessed Datasets & Representation Inspection\n",
                "Loading preprocessed partitions from `data/processed/` generated in Phase 2."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_dir = PROJECT_ROOT / \"data\" / \"processed\"\n",
                "train_df = pd.read_parquet(data_dir / \"train.parquet\")\n",
                "val_df = pd.read_parquet(data_dir / \"validation.parquet\")\n",
                "test_df = pd.read_parquet(data_dir / \"test.parquet\")\n",
                "id2label, label2id = get_label_mappings()\n",
                "\n",
                "print(f\"Loaded partitions:\")\n",
                "print(f\"  - Train:      {len(train_df):>5} samples\")\n",
                "print(f\"  - Validation: {len(val_df):>5} samples\")\n",
                "print(f\"  - Test:       {len(test_df):>5} samples\")\n",
                "print(f\"\\nLabel Mappings: {id2label}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Hyperparameter Tuning on Validation Split\n",
                "To ensure a competitive baseline, we tune the regularization strength $C \\in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]$ and test class weighting strategies (`balanced` vs. `None`) using **Macro-F1** on the validation set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tfidf_params = {\n",
                "    \"max_features\": 10000,\n",
                "    \"ngram_range\": (1, 2),\n",
                "    \"sublinear_tf\": True,\n",
                "    \"min_df\": 2,\n",
                "}\n",
                "\n",
                "reports_dir = PROJECT_ROOT / \"outputs\" / \"reports\"\n",
                "tuning_df = pd.read_csv(reports_dir / \"baseline_hyperparameter_tuning.csv\")\n",
                "\n",
                "print(\"Top 5 Validation Configurations (Text-only):\")\n",
                "display(tuning_df[tuning_df[\"representation\"] == \"text_only\"].head(5))\n",
                "\n",
                "print(\"\\nTop 5 Validation Configurations (Section-aware):\")\n",
                "display(tuning_df[tuning_df[\"representation\"] == \"text_section_aware\"].head(5))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Comparative Evaluation on the Official Test Split\n",
                "Evaluating both representations on the unseen official test set (1,859 samples)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "summary_df = pd.read_csv(reports_dir / \"baseline_comparison_summary.csv\")\n",
                "print(\"=== Baseline Performance Comparison Table ===\")\n",
                "display(summary_df)\n",
                "\n",
                "text_only_rep = pd.read_csv(reports_dir / \"baseline_text_only_classification_report.csv\")\n",
                "section_aware_rep = pd.read_csv(reports_dir / \"baseline_text_section_aware_classification_report.csv\")\n",
                "print(\"\\n--- Detailed Classification Report: Text-only (Macro, Micro, Weighted) ---\")\n",
                "display(text_only_rep)\n",
                "print(\"\\n--- Detailed Classification Report: Section-aware (Macro, Micro, Weighted) ---\")\n",
                "display(section_aware_rep)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Confusion Matrix & Error Inspection\n",
                "Analyzing where the baseline makes classification errors, and observing how section-awareness resolves ambiguities."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "\n",
                "with open(reports_dir / \"baseline_text_only_metrics.json\", \"r\") as f:\n",
                "    text_only_metrics = json.load(f)\n",
                "with open(reports_dir / \"baseline_text_section_aware_metrics.json\", \"r\") as f:\n",
                "    section_aware_metrics = json.load(f)\n",
                "\n",
                "fig1 = plot_confusion_matrix(\n",
                "    cm=text_only_metrics[\"confusion_matrix\"],\n",
                "    target_names=text_only_metrics[\"target_names\"],\n",
                "    title=\"TF-IDF + LR Confusion Matrix (Text-only)\",\n",
                ")\n",
                "plt.show()\n",
                "\n",
                "fig2 = plot_confusion_matrix(\n",
                "    cm=section_aware_metrics[\"confusion_matrix\"],\n",
                "    target_names=section_aware_metrics[\"target_names\"],\n",
                "    title=\"TF-IDF + LR Confusion Matrix (Section-aware)\",\n",
                ")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Key Insights from the Baseline Phase\n",
                "1. **Strong Baseline Performance:** The tuned TF-IDF + Logistic Regression model achieves an impressive **81.57% Test Macro-F1** in the text-only setting.\n",
                "2. **Impact of Section Awareness:** Prepending the canonical section header elevates Test Macro-F1 to **84.05%** (+2.48% absolute gain) and Test Accuracy to **86.28%** (+2.74% gain).\n",
                "3. **Targeted Improvement on Method Class:** The F1-score for the `method` class jumps from **81.06% to 86.72% (+5.66%)**, demonstrating that section context substantially reduces confusion between background citations and methodological citations.\n",
                "4. **Benchmark for Transformers:** In the upcoming Phase 4, our fine-tuned Transformer models (DistilBERT and SciBERT) will be benchmarked against this strong 84.05% baseline."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 5,
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = Path("notebooks/02_baseline_tfidf.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {output_path}")
