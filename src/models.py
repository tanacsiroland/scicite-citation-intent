"""Model architectures and training pipelines for citation intent classification."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
)

from src.data import ID2LABEL, LABEL2ID, CLASS_NAMES
from src.utils import compute_classification_metrics


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text_only",
    seed: int = 42,
    c_grid: Optional[List[float]] = None,
) -> Tuple[Pipeline, np.ndarray, Dict[str, Any]]:
    """Train TF-IDF + Logistic Regression baseline with validation tuning."""
    if c_grid is None:
        c_grid = [0.5, 1.0, 2.0, 5.0]

    best_macro_f1 = -1.0
    best_c = 1.0
    best_cw = "balanced"

    tfidf_kwargs = {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
    }

    # Quick tuning on validation set
    for cw in ["balanced", None]:
        for c_val in c_grid:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
                ("clf", LogisticRegression(C=c_val, class_weight=cw, solver="lbfgs", max_iter=1000, random_state=seed)),
            ])
            pipe.fit(train_df[text_column], train_df["label"])
            val_preds = pipe.predict(val_df[text_column])
            val_m = compute_classification_metrics(val_df["label"], val_preds)
            if val_m["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_m["macro_f1"]
                best_c = c_val
                best_cw = cw

    # Final fit on train with best params
    optimal_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
        ("clf", LogisticRegression(C=best_c, class_weight=best_cw, solver="lbfgs", max_iter=1000, random_state=seed)),
    ])
    optimal_pipeline.fit(train_df[text_column], train_df["label"])
    test_preds = optimal_pipeline.predict(test_df[text_column])
    metrics = compute_classification_metrics(test_df["label"], test_preds)
    metrics["best_params"] = {"C": best_c, "class_weight": best_cw}

    return optimal_pipeline, test_preds, metrics


def train_transformer(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    output_dir: Path,
    seed: int = 42,
    num_epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 512,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """Fine-tune a pretrained Hugging Face Transformer and evaluate on test set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def _tokenize(df: pd.DataFrame):
        enc = tokenizer(
            df[text_column].fillna("").astype(str).tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return Dataset.from_dict({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": df["label"].tolist(),
        })

    train_ds = _tokenize(train_df)
    val_ds = _tokenize(val_df)
    test_ds = _tokenize(test_df)

    def _compute_metrics(eval_pred: EvalPrediction):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        res = compute_classification_metrics(eval_pred.label_ids, preds)
        return {"eval_accuracy": res["accuracy"], "eval_macro_f1": res["macro_f1"]}

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_steps=100,
        bf16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    pred_output = trainer.predict(test_ds)
    test_preds = np.argmax(pred_output.predictions, axis=-1)
    metrics = compute_classification_metrics(test_df["label"], test_preds)

    return trainer, test_preds, metrics
