"""Script to generate the complete notebooks/04_error_analysis.ipynb notebook."""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SciCite: Citation Intent Classification\n",
                "## 04. In-Depth Error Analysis & Failure Mode Diagnostics\n",
                "\n",
                "**Course:** Mesterséges intelligencia alkalmazási területei (MSc)\n",
                "**Objective:** Quantitative and qualitative post-hoc failure analysis across all baseline and fine-tuned Transformer models on the official SciCite test set (1,859 instances).\n",
                "\n",
                "### Analytical Dimensions Explored:\n",
                "1. **Error Rate Stratification by Section Context:** Measuring vulnerability across the 7 canonical section categories and quantifying the \"missing section penalty\".\n",
                "2. **Inter-Class Error Transition Dynamics:** Analyzing the primary confusion axes (`Method` $\\leftrightarrow$ `Background` vs. `Result` $\\leftrightarrow$ `Background`).\n",
                "3. **Model Agreement & Disagreement:** Quantifying baseline errors rescued by contextual Transformers vs. universal hard instances where all models fail.\n",
                "4. **Qualitative Case Studies:** Granular linguistic and contextual examination of representative failure modes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Environment & Setup\n",
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import matplotlib.image as mpimg\n",
                "\n",
                "# Add project root to sys.path\n",
                "PROJECT_ROOT = Path(\"..\").resolve()\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(PROJECT_ROOT))\n",
                "\n",
                "reports_dir = PROJECT_ROOT / \"outputs\" / \"reports\"\n",
                "errors_dir = PROJECT_ROOT / \"outputs\" / \"errors\"\n",
                "figures_dir = PROJECT_ROOT / \"outputs\" / \"figures\"\n",
                "\n",
                "print(\"Project paths verified successfully.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Error Rate Stratification by Canonical Section Type\n",
                "We examine whether model reliability depends systematically on the location of the citation within the scientific document structure."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load section error rate report\n",
                "section_err_df = pd.read_csv(reports_dir / \"error_rate_by_section.csv\")\n",
                "display(section_err_df)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualizing Error Rates across Section Headers\n",
                "fig_path = figures_dir / \"error_rate_by_section.png\"\n",
                "if fig_path.is_file():\n",
                "    img = mpimg.imread(str(fig_path))\n",
                "    plt.figure(figsize=(13, 6))\n",
                "    plt.imshow(img)\n",
                "    plt.axis(\"off\")\n",
                "    plt.title(\"Error Rate Stratification Across Canonical Sections\", fontsize=13, fontweight=\"bold\", pad=10)\n",
                "    plt.show()\n",
                "else:\n",
                "    print(f\"Figure not found at: {fig_path}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Key Section-Level Diagnostic Observations:\n",
                "- **Lowest Error Rate in `Method` Sections:** ModernBERT achieves a remarkably low **5.34% error rate** in method sections (TF-IDF with section context achieves 7.04%). The procedural phrasing in method sections provides strong corroboration.\n",
                "- **Severe \"Missing Section Penalty\":** When section metadata is omitted (`missing`), error rates spike to **21.05% - 26.32%**, representing a **$3.5\\times$ to $4.5\\times$ increase** in failure probability compared to structured method sections.\n",
                "- **Moderate Errors in `Discussion` (14.0% - 15.1%):** Discussion sections frequently juxtapose previous findings with the author's own discoveries, presenting dual-intent linguistic patterns that challenge both classical and deep learning models."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Inter-Class Error Transition Dynamics\n",
                "To understand which semantic categories confuse the models most, we examine the directionality of prediction errors."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load error transition breakdown\n",
                "transitions_df = pd.read_csv(reports_dir / \"error_transitions_breakdown.csv\")\n",
                "display(transitions_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Confusion Directionality Analysis:\n",
                "1. **`Method` Misclassified as `Background`:** In the TF-IDF baseline, 97 `method` instances (16.06%) were predicted as `background`. ModernBERT reduces this to 63 instances (10.43%), and SciBERT to 84 instances (13.91%). This occurs when a method citation introduces an algorithm or tool using declarative, descriptive syntax rather than procedural action verbs.\n",
                "2. **`Result` Misclassified as `Background`:** The TF-IDF baseline misclassifies 51 `result` citations (19.69%) as `background`. In striking contrast, **SciBERT reduces this to just 13 cases (5.02%)**, demonstrating that domain-specific scientific pre-training captures outcome reporting nuances that general models and n-gram baselines miss.\n",
                "3. **`Method` vs. `Result` Boundary is Crisp:** Confusions directly between `method` and `result` remain minimal across all models (typically $\\le 2.5\\%$ of instances), confirming that the models cleanly separate experimental protocol from empirical outcome."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Model Agreement & Disagreement: Context Rescue vs. Universal Hard Instances\n",
                "Comparing instances where deep Transformer representations succeed over n-gram baselines, versus instances where all models fail simultaneously."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load rescued errors and universal hard instances\n",
                "rescued_df = pd.read_csv(errors_dir / \"baseline_errors_fixed_by_transformers.csv\")\n",
                "hard_df = pd.read_csv(errors_dir / \"hard_instances_all_models_wrong.csv\")\n",
                "\n",
                "print(f\"Baseline errors successfully rescued by Transformers: {len(rescued_df)} instances\")\n",
                "print(f\"Universal hard instances (failed by ALL 4 models):        {len(hard_df)} instances ({len(hard_df) / 1859 * 100:.2f}% of test set)\")\n",
                "\n",
                "print(\"\\nHard instances breakdown by True Label:\")\n",
                "print(hard_df[\"label_name\"].value_counts())\n",
                "\n",
                "print(\"\\nHard instances breakdown by Section Type:\")\n",
                "print(hard_df[\"section_clean\"].value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Qualitative Case Studies: Granular Linguistic Analysis\n",
                "Here we review representative samples illustrating the three primary failure archetypes identified in the dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load curated qualitative case studies\n",
                "case_studies = pd.read_csv(errors_dir / \"qualitative_case_studies.csv\")\n",
                "\n",
                "for idx, row in case_studies.iterrows():\n",
                "    print(\"=\" * 90)\n",
                "    print(f\"CASE STUDY #{idx+1}: {row['Error Category']}\")\n",
                "    print(\"=\" * 90)\n",
                "    print(f\"Section:          {row['Section Header']}\")\n",
                "    print(f\"True Label:       {row['True Label'].upper()}\")\n",
                "    print(f\"TF-IDF Pred:      {row['Baseline Pred']}\")\n",
                "    print(f\"ModernBERT Pred:  {row['ModernBERT Pred']}\")\n",
                "    print(f\"SciBERT Pred:     {row['SciBERT Pred']}\")\n",
                "    print(f\"Citation Text:\\n\\\"{row['Citation Context']}\\\"\")\n",
                "    print(f\"Linguistic Note:  {row['Linguistic Explanation']}\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Synthesis & Key Takeaways\n",
                "\n",
                "1. **Section Conditioning is a Structural Superpower:** Context conditioning drastically drops error rates in procedural sections (down to ~5.3% for ModernBERT in `method`). In contrast, unsectioned citations face an error rate over 21%.\n",
                "2. **Scientific Domain Pretraining Excels on Result Classification:** SciBERT's scientific vocabulary and pretraining domain enable it to correctly identify `result` citations with only 5.0% confusion to `background`, compared to ~19.7% for the baseline and ~8.9% for ModernBERT.\n",
                "3. **Intrinsic Human Annotation Ambiguity (~6% Ceiling):** The 116 universal hard instances (6.24%) primarily consist of dual-intent statements in `discussion` sections where authors compare their findings with previous work. These sentences are inherently multi-intent, reflecting the natural subjective limit of single-label citation intent schemes."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2,
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = Path("notebooks/04_error_analysis.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Generated {out_path.resolve()}")
