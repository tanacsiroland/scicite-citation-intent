"""Baseline TF-IDF + Logistic Regression classification model."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.metrics import compute_metrics


class TfidfBaseline:
    """End-to-end Pipeline combining TF-IDF vectorization with Logistic Regression."""

    def __init__(
        self,
        tfidf_params: Optional[Dict[str, Any]] = None,
        lr_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        """Initialize the TF-IDF and Logistic Regression pipeline.

        Args:
            tfidf_params: Parameters for TfidfVectorizer (e.g., max_features, ngram_range).
            lr_params: Parameters for LogisticRegression (e.g., C, solver, class_weight).
            random_state: Seed for deterministic solver convergence.
        """
        default_tfidf = {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "sublinear_tf": True,
            "min_df": 2,
        }
        default_lr = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": random_state,
        }

        self.tfidf_params = {**default_tfidf, **(tfidf_params or {})}
        if isinstance(self.tfidf_params.get("ngram_range"), list):
            self.tfidf_params["ngram_range"] = tuple(self.tfidf_params["ngram_range"])

        self.lr_params = {**default_lr, **(lr_params or {})}
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**self.tfidf_params)),
            ("clf", LogisticRegression(**self.lr_params)),
        ])

    def fit(self, X: pd.Series | List[str], y: pd.Series | np.ndarray) -> "TfidfBaseline":
        """Fit the TF-IDF vectorizer and train the Logistic Regression classifier.

        Args:
            X: Input text series (citation context).
            y: Integer class labels.

        Returns:
            Fitted instance of self.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.Series | List[str]) -> np.ndarray:
        """Predict integer class labels for input texts."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.Series | List[str]) -> np.ndarray:
        """Predict class probability distribution for input texts."""
        return self.pipeline.predict_proba(X)

    def evaluate(
        self,
        X: pd.Series | List[str],
        y: pd.Series | np.ndarray,
        id2label: Dict[int, str],
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics on evaluation data.

        Args:
            X: Input text series.
            y: Ground truth integer class labels.
            id2label: Integer to class string mapping.

        Returns:
            Dictionary containing accuracy, macro_f1, per-class breakdown, and confusion matrix.
        """
        y_pred = self.predict(X)
        metrics = compute_metrics(y_true=y, y_pred=y_pred, id2label=id2label)
        return metrics


def tune_baseline_hyperparameters(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    id2label: Dict[int, str],
    c_grid: Optional[List[float]] = None,
    class_weights: Optional[List[Optional[str]]] = None,
    tfidf_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Perform validation-set grid search over regularization C and class weights.

    Uses validation Macro-F1 as the primary optimization objective.

    Args:
        X_train: Training texts.
        y_train: Training labels.
        X_val: Validation texts.
        y_val: Validation labels.
        id2label: Label ID to name mapping.
        c_grid: List of regularization values to evaluate.
        class_weights: List of class_weight options (e.g. [None, 'balanced']).
        tfidf_params: Static TF-IDF parameter dictionary.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (best_params_dict, grid_search_results_dataframe).
    """
    if c_grid is None:
        c_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    if class_weights is None:
        class_weights = ["balanced", None]

    records = []
    best_macro_f1 = -1.0
    best_params = {}

    for cw in class_weights:
        for c_val in c_grid:
            lr_params = {
                "C": c_val,
                "class_weight": cw,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": random_state,
            }
            model = TfidfBaseline(
                tfidf_params=tfidf_params,
                lr_params=lr_params,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            val_metrics = model.evaluate(X_val, y_val, id2label=id2label)

            macro_f1 = val_metrics["macro_f1"]
            accuracy = val_metrics["accuracy"]

            records.append({
                "C": c_val,
                "class_weight": "balanced" if cw == "balanced" else "none",
                "val_macro_f1": macro_f1,
                "val_accuracy": accuracy,
                "val_result_f1": val_metrics["per_class"][2]["f1"],
            })

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_params = {"C": c_val, "class_weight": cw}

    results_df = pd.DataFrame(records).sort_values("val_macro_f1", ascending=False)
    return best_params, results_df
