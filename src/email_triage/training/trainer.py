"""Training loop with MLflow tracking for multi-class email triage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from email_triage.labels import CLASS_NAMES
from email_triage.training.metrics import EvalResult, compute_metrics


@dataclass
class TrainConfig:
    """Hyperparameters and paths for a training run."""

    model_name: str = "distilbert-base-uncased"
    lr: float = 2e-5
    epochs: int = 10
    batch_size: int = 16
    max_grad_norm: float = 1.0
    data_dir: str = "data/processed"
    checkpoint_path: str = "models/best.pt"
    report_path: str = "outputs/eval_report.txt"
    experiment_name: str = "email-triage"
    tracking_uri: str = "sqlite:///outputs/mlflow.db"


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    max_grad_norm: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one epoch of training or evaluation.

    Args:
        model: The classifier model.
        loader: DataLoader for the split.
        criterion: Loss function (CrossEntropyLoss).
        device: Torch device.
        optimizer: If provided, runs in train mode; otherwise eval mode.
        max_grad_norm: Gradient clipping norm.

    Returns:
        (avg_loss, y_true, y_probs) where y_true is 1D int array (N,)
        and y_probs is 2D float array (N, num_classes).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_true: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    context = torch.no_grad() if not is_train else torch.enable_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_true.append(labels.cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_true)
    y_probs = np.concatenate(all_probs)
    return avg_loss, y_true, y_probs


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, EvalResult]:
    """Evaluate the model on a dataset split.

    Returns:
        (avg_loss, EvalResult).
    """
    loss, y_true, y_probs = _run_epoch(model, loader, criterion, device)
    y_pred = y_probs.argmax(axis=-1)
    return loss, compute_metrics(y_true, y_pred)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device | None = None,
) -> nn.Module:
    """Full training loop with MLflow tracking and checkpointing.

    Args:
        model: The EmailTriageClassifier.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration.
        device: Torch device. Auto-detected if None.

    Returns:
        The trained model (loaded from best checkpoint).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr)

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    best_macro_f1 = -1.0

    with mlflow.start_run():
        mlflow.log_params(asdict(config))

        for epoch in range(1, config.epochs + 1):
            # --- Train ---
            train_loss, _, _ = _run_epoch(
                model, train_loader, criterion, device,
                optimizer=optimizer, max_grad_norm=config.max_grad_norm,
            )

            # --- Validate ---
            val_loss, val_result = evaluate(
                model, val_loader, criterion, device,
            )

            # --- Log to MLflow ---
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_result.macro_f1,
                    "val_accuracy": val_result.accuracy,
                },
                step=epoch,
            )
            for class_name in CLASS_NAMES:
                mlflow.log_metric(
                    f"val_f1_{class_name}",
                    val_result.per_class[class_name]["f1"],
                    step=epoch,
                )

            # --- Console output ---
            print(
                f"Epoch {epoch}/{config.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_macro_f1={val_result.macro_f1:.4f}  "
                f"val_accuracy={val_result.accuracy:.4f}"
            )

            # --- Checkpoint best model ---
            if val_result.macro_f1 > best_macro_f1:
                best_macro_f1 = val_result.macro_f1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(config),
                        "val_macro_f1": best_macro_f1,
                    },
                    checkpoint_path,
                )
                print(f"  -> Saved best model (macro_f1={best_macro_f1:.4f})")

        mlflow.log_metric("best_val_macro_f1", best_macro_f1)

    # Reload best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nTraining complete. Best val macro_f1={best_macro_f1:.4f} (epoch {checkpoint['epoch']})")

    return model
