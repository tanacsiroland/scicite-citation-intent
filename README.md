# SciCite: Citation Intent Classification

Automated classification of citation intents in scientific literature using the official `allenai/scicite` benchmark dataset into three target categories: `method`, `background`, and `result`.

This project evaluates:
1. A classical **TF-IDF + Logistic Regression baseline** with hyperparameter tuning.
2. Fine-tuned general-domain Transformers (**DistilBERT**, **ModernBERT**) and domain-adapted scientific Transformers (**SciBERT**).
3. The impact of incorporating academic section context (`sectionName`) via structured prefix conditioning (**Text-only vs. Section-aware**).

---

## Benchmark Leaderboard (Official Test Split, N=1,859)

| Model | Input Mode | Accuracy | Macro F1 | Method F1 | Background F1 | Result F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ModernBERT** | **Section-aware** | **87.25%** | **0.8550** | **0.8925** | **0.8868** | 0.7858 |
| **SciBERT** | Text-only | 86.18% | 0.8503 | 0.8559 | 0.8780 | **0.8171** |
| **DistilBERT** | Section-aware | 85.64% | 0.8433 | 0.8586 | 0.8727 | 0.7987 |
| **Baseline (TF-IDF + LR)** | Section-aware | 86.28% | 0.8405 | 0.8672 | 0.8844 | 0.7698 |
| **ModernBERT** | Text-only | 85.64% | 0.8404 | 0.8540 | 0.8764 | 0.7907 |
| **DistilBERT** | Text-only | 85.48% | 0.8397 | 0.8487 | 0.8770 | 0.7933 |
| **SciBERT** | Section-aware | 85.31% | 0.8392 | 0.8615 | 0.8691 | 0.7870 |
| **Baseline (TF-IDF + LR)** | Text-only | 83.54% | 0.8157 | 0.8106 | 0.8643 | 0.7723 |

### Key Takeaways:
- **Structural Section Conditioning:** Adding normalized section headers establishes a strong prior, improving the TF-IDF baseline by **+2.5% Macro-F1** and driving ModernBERT to the overall top score (**87.25% Accuracy, 0.8550 Macro-F1**).
- **Domain Adaptation (SciBERT):** Pre-trained on 1.14M scientific publications with domain-specific vocabulary (`scivocab`), SciBERT excels on the most challenging minority class, reaching **0.8171 Result F1**.
- **Overfitting Dynamics:** Validation loss reaches its global minimum at epoch 2–3 across all architectures, validating the Devlin et al. (2018) recommendation of 3–4 fine-tuning epochs.

---

## Repository Structure

```text
scicite-citation-intent/
├── citation_intent_classification.ipynb  # Interactive end-to-end master notebook
├── train_all.py                          # Unified CLI pipeline runner
├── README.md                             # Project overview and leaderboard
├── requirements.txt                      # Clean minimal dependencies
├── src/                                  # Modular core library
│   ├── __init__.py
│   ├── data.py                           # Dataset loading, normalization & representations
│   ├── models.py                         # Baseline & Transformer training routines
│   └── utils.py                          # Metrics, seed & plotting utilities
├── outputs/
│   ├── figures/                          # 5 publication-grade figures
│   │   ├── class_distribution.png
│   │   ├── canonical_section_contingency.png
│   │   ├── transformer_training_curves.png
│   │   ├── confusion_matrices_comparison.png
│   │   └── error_rate_by_section.png
│   ├── all_models_comparative_summary.csv # Master metrics summary table
│   ├── qualitative_case_studies.csv       # 6 qualitative case studies
│   └── test_predictions.csv              # Test predictions across all models
└── data/
    ├── raw/scicite/                      # dev.jsonl, test.jsonl, train.jsonl
    └── processed/                        # train, validation, test (.parquet)
```

---

## Quickstart & Reproducibility

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Interactive Master Notebook
Open `citation_intent_classification.ipynb` in Jupyter Lab, Jupyter Notebook, or VS Code. The notebook is fully pre-executed with embedded figures and tables, and executes end-to-end in seconds.

### 3. Command-Line Pipeline Execution
```bash
# Run baseline evaluation on preprocessed splits:
python train_all.py

# Run full Transformer fine-tuning (GPU recommended):
python train_all.py --train-transformers
```
