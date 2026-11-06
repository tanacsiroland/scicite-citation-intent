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
├── 01_results_and_evaluation.ipynb   # Executive presentation, benchmark leaderboard & error analysis
├── 02_pipeline_training.ipynb        # Complete end-to-end training pipeline & model saving
├── 03_interactive_inference.ipynb    # Zero-setup interactive demo & test playground for evaluators
├── README.md                         # Project overview, benchmark leaderboard & quickstart
├── requirements.txt                  # Minimal clean dependencies
├── src/                              # Modular core library
│   ├── __init__.py
│   ├── data.py                       # Dataset loading, normalization & representations
│   ├── models.py                     # Baseline & Transformer training routines + HF publishing
│   └── utils.py                      # Metrics, live plotting & inference visualizer
├── outputs/
│   ├── figures/                      # 5 publication-grade figures
│   │   ├── class_distribution.png
│   │   ├── canonical_section_contingency.png
│   │   ├── transformer_training_curves.png
│   │   ├── confusion_matrices_comparison.png
│   │   └── error_rate_by_section.png
│   ├── models/                       # Locally saved model artifacts & baseline pipeline
│   │   ├── baseline_tfidf.pkl
│   │   ├── scicite-modernbert-section-aware/
│   │   ├── scicite-modernbert-text-only/
│   │   ├── scicite-scibert-section-aware/
│   │   ├── scicite-scibert-text-only/
│   │   ├── scicite-distilbert-section-aware/
│   │   └── scicite-distilbert-text-only/
│   ├── all_models_comparative_summary.csv # Master metrics summary table
│   ├── qualitative_case_studies.csv       # 6 qualitative case studies
│   └── test_predictions.csv              # Test predictions across all models
└── data/
    ├── raw/scicite/                  # dev.jsonl, test.jsonl, train.jsonl
    └── processed/                    # train, validation, test (.parquet)
```

---

## Hosted Models on Hugging Face Hub (Public)

All **6 fine-tuned Transformer configurations** have been published to the public Hugging Face Hub under the [`Tancsi`](https://huggingface.co/Tancsi) profile for zero-credential access:

| Model Architecture | Input Mode | Test Accuracy | Macro F1 | Hugging Face Repository |
| :--- | :--- | :---: | :---: | :--- |
| **ModernBERT** | **Section-aware** | **87.25%** | **0.8550** | [`Tancsi/scicite-modernbert-section-aware`](https://huggingface.co/Tancsi/scicite-modernbert-section-aware) *(Overall Champion)* |
| **SciBERT** | **Text-only** | **86.18%** | **0.8503** | [`Tancsi/scicite-scibert-text-only`](https://huggingface.co/Tancsi/scicite-scibert-text-only) *(Best on Result Class: 0.8171 F1)* |
| **DistilBERT** | **Section-aware** | **85.64%** | **0.8433** | [`Tancsi/scicite-distilbert-section-aware`](https://huggingface.co/Tancsi/scicite-distilbert-section-aware) |
| **ModernBERT** | **Text-only** | **85.64%** | **0.8404** | [`Tancsi/scicite-modernbert-text-only`](https://huggingface.co/Tancsi/scicite-modernbert-text-only) |
| **SciBERT** | **Section-aware** | **85.31%** | **0.8392** | [`Tancsi/scicite-scibert-section-aware`](https://huggingface.co/Tancsi/scicite-scibert-section-aware) |
| **DistilBERT** | **Text-only** | **85.48%** | **0.8397** | [`Tancsi/scicite-distilbert-text-only`](https://huggingface.co/Tancsi/scicite-distilbert-text-only) |

Additionally, the fitted classical **TF-IDF + Logistic Regression pipeline** is saved locally at `outputs/models/baseline_tfidf.pkl`.

---

## Quickstart & Evaluation Guide

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Notebook Guide

| Notebook | Purpose | Runtime | Prerequisites |
| :--- | :--- | :---: | :--- |
| **`01_results_and_evaluation.ipynb`** | **Executive Defense**: Inspect pre-computed leaderboard, metrics, figures, training dynamics, and linguistic error case studies. | ~5 sec | None |
| **`02_pipeline_training.ipynb`** | **Full Reproducibility**: Run the entire pipeline from scratch (data loading, live dynamic plotting, baseline tuning, transformer training, local model saving, and optional HF publishing). | ~2 min (GPU) | Local GPU / Colab |
| **`03_interactive_inference.ipynb`** | **Lecturer Playground**: Interactive cell to input custom academic sentences and test predictions with probability bar charts. | ~10 sec | None (pulls from public HF Hub) |

### 3. Direct Python Inference (2 Lines of Code)
```python
from transformers import pipeline

# Load directly from public Hugging Face repository (no token or login required)
classifier = pipeline("text-classification", model="Tancsi/scicite-modernbert-section-aware", top_k=None)
result = classifier("Section: method. Citation: We trained the model using AdamW with learning rate 2e-5.")
print(result)
```
