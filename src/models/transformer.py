"""Transformer fine-tuning and evaluation module for citation intent classification."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    DataCollatorWithPadding,
)

from src.utils.metrics import compute_metrics


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer: Any,
    text_column: str,
    max_length: int = 512,
) -> Tuple[Dataset, int]:
    """Tokenize a pandas DataFrame into a Hugging Face Dataset.

    Also calculates the exact number and percentage of sequences that hit
    the truncation ceiling (max_length).

    Args:
        df: Input DataFrame containing text and labels.
        tokenizer: Hugging Face PreTrainedTokenizer.
        text_column: Name of column containing input text strings.
        max_length: Maximum token length ceiling. Defaults to 512.

    Returns:
        Tuple of (Hugging Face Dataset, count_of_truncated_sequences).
    """
    texts = df[text_column].fillna("").astype(str).tolist()
    labels = df["label"].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding performed by DataCollator
        return_attention_mask=True,
    )

    # Compute truncation frequency: length of tokenized input_ids == max_length
    truncated_count = sum(len(ids) >= max_length for ids in encodings["input_ids"])

    data_dict = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "label": labels,
    }

    hf_dataset = Dataset.from_dict(data_dict)
    return hf_dataset, truncated_count


def make_compute_metrics_fn(id2label: Dict[int, str]):
    """Create a metric computation function compatible with Hugging Face Trainer.

    Args:
        id2label: Integer to class string mapping.

    Returns:
        Callable taking EvalPrediction and returning evaluation metrics dictionary.
    """
    def compute_eval_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        res = compute_metrics(y_true=labels, y_pred=preds, id2label=id2label)
        return {
            "eval_accuracy": res["accuracy"],
            "eval_macro_f1": res["macro_f1"],
            "eval_macro_precision": res["macro_precision"],
            "eval_macro_recall": res["macro_recall"],
            "eval_micro_f1": res["micro_f1"],
            "eval_weighted_f1": res["weighted_f1"],
        }

    return compute_eval_metrics


def train_and_evaluate_transformer(
    model_name_or_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    output_dir: Path,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    training_args_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fine-tune a pretrained Transformer model and evaluate on test data.

    Args:
        model_name_or_path: Pretrained Hugging Face model identifier.
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        text_column: Text column name ('text_only' or 'text_section_aware').
        output_dir: Directory where model checkpoints and logs are saved.
        id2label: Dict mapping class ID to class string.
        label2id: Dict mapping class string to class ID.
        training_args_kwargs: Additional or overriding arguments for TrainingArguments.
        seed: Random seed for initialization.

    Returns:
        Dictionary containing test metrics, predictions, probabilities, and truncation statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer & model: {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    max_length = training_args_kwargs.get("max_length", 512) if training_args_kwargs else 512

    # Tokenize splits
    train_dataset, train_trunc = tokenize_dataframe(train_df, tokenizer, text_column, max_length)
    val_dataset, val_trunc = tokenize_dataframe(val_df, tokenizer, text_column, max_length)
    test_dataset, test_trunc = tokenize_dataframe(test_df, tokenizer, text_column, max_length)

    print(f"  Truncation check (max_length={max_length}):")
    print(f"    Train: {train_trunc} / {len(train_df)} ({train_trunc / len(train_df) * 100:.2f}%)")
    print(f"    Val:   {val_trunc} / {len(val_df)} ({val_trunc / len(val_df) * 100:.2f}%)")
    print(f"    Test:  {test_trunc} / {len(test_df)} ({test_trunc / len(test_df) * 100:.2f}%)")

    default_training_args = {
        "output_dir": str(output_dir),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 4,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "bf16": torch.cuda.is_available(),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_steps": 50,
        "report_to": "none",
        "seed": seed,
    }

    if training_args_kwargs:
        # Filter out custom non-TrainingArguments parameters like max_length
        cleaned_kwargs = {k: v for k, v in training_args_kwargs.items() if k not in ["max_length"]}
        default_training_args.update(cleaned_kwargs)

    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    valid_args = {k: v for k, v in default_training_args.items() if k in sig.parameters}
    training_args = TrainingArguments(**valid_args)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_eval_metrics = make_compute_metrics_fn(id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_eval_metrics,
    )

    print(f"Starting training for {training_args.num_train_epochs} epochs on device: {training_args.device} (bf16={training_args.bf16})...")
    train_result = trainer.train()

    print("Running evaluation on official unseen Test split...")
    test_pred_output = trainer.predict(test_dataset)
    logits = test_pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)

    y_test = test_df["label"].to_numpy()
    test_metrics = compute_metrics(y_true=y_test, y_pred=preds, id2label=id2label)

    return {
        "test_metrics": test_metrics,
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "log_history": trainer.state.log_history,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "truncation": {
            "train": train_trunc,
            "val": val_trunc,
            "test": test_trunc,
        },
    }
