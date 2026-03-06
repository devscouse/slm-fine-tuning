"""Train the email triage classifier and evaluate on the test set."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from email_triage.data.dataset import EmailDataset
from email_triage.model.classifier import build_classifier, build_tokenizer
from email_triage.training.trainer import TrainConfig, evaluate, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the email triage classifier.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint-path", default="models/best.pt")
    parser.add_argument("--report-path", default="outputs/eval_report.txt")
    parser.add_argument("--experiment-name", default="email-triage")
    parser.add_argument("--tracking-uri", default="sqlite:///outputs/mlflow.db")
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model_name,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        report_path=args.report_path,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
    )

    data_dir = Path(config.data_dir)
    tokenizer = build_tokenizer(config.model_name)
    model = build_classifier(config.model_name)

    train_ds = EmailDataset(data_dir / "train.jsonl", tokenizer)
    val_ds = EmailDataset(data_dir / "val.jsonl", tokenizer)
    test_ds = EmailDataset(data_dir / "test.jsonl", tokenizer)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = train(model, train_loader, val_loader, config, device)

    # --- Final test evaluation ---
    print("\n--- Test Set Evaluation ---")
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_result = evaluate(model, test_loader, criterion, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test macro_f1: {test_result.macro_f1:.4f}")
    print(f"Test accuracy: {test_result.accuracy:.4f}")
    print(f"\n{test_result.report_text}")

    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(test_result.report_text, encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
