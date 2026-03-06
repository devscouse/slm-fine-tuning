"""Training module for email triage classifier."""

from email_triage.training.metrics import EvalResult, compute_metrics
from email_triage.training.trainer import TrainConfig, evaluate, train

__all__ = ["EvalResult", "TrainConfig", "compute_metrics", "evaluate", "train"]
