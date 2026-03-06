"""Evaluation metrics for multi-class email triage classification."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from email_triage.labels import CLASS_NAMES, NUM_CLASSES


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    macro_f1: float
    accuracy: float
    per_class: dict[str, dict[str, float]]
    report_text: str


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    """Compute multi-class classification metrics.

    Args:
        y_true: Ground-truth class indices, shape (N,).
        y_pred: Predicted class indices, shape (N,).

    Returns:
        EvalResult with macro F1, accuracy, per-class metrics, and text report.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    all_labels = list(range(NUM_CLASSES))

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "f1": float(f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
        }

    report_text = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    return EvalResult(
        macro_f1=macro_f1,
        accuracy=accuracy,
        per_class=per_class,
        report_text=report_text,
    )
