# SLM Fine-Tuning — Email Triage

A learning project for fine-tuning a small language model (SLM) on a
**multi-label classification** task: given the subject and body of an email,
predict all applicable triage labels simultaneously.

## Labels

| Label | Description |
|---|---|
| `urgent` | Time-sensitive; needs attention today or risk of a missed deadline / escalation |
| `needs_reply` | Sender is explicitly expecting or waiting for a response |
| `action_required` | Recipient must take a concrete action (approve, sign, submit, fix, etc.) |
| `order_confirmation` | Confirms a purchase, shipment, or subscription sign-up |
| `alerts` | Automated system or service alert (security notice, outage, threshold breach) |
| `calendar_event` | Meeting invite, scheduling request, RSVP, or calendar update |
| `newsletters` | Newsletter, marketing digest, promotional email, or mailing-list post |

An email can carry **any combination** of these labels.

## Project Structure

```
slm-fine-tuning/
├── data/
│   ├── raw/            # Source datasets (e.g. Enron, TREC)
│   ├── synthetic/      # LLM-generated labelled examples
│   └── processed/      # Tokenized, model-ready splits
│
├── src/email_triage/
│   ├── labels.py       # Label taxonomy and vector helpers
│   ├── data/           # Dataset loading and preprocessing
│   ├── model/          # Model definition and config
│   ├── training/       # Training loop and metrics
│   └── inference/      # Prediction pipeline
│
├── notebooks/          # Exploratory notebooks (data, training, eval)
├── models/             # Saved checkpoints
├── outputs/            # Training logs and results
└── tests/
```

## Pipeline Overview

1. **Data** — collect raw emails, generate synthetic labelled examples, preprocess into train/val/test splits
2. **Model** — load a pretrained SLM (e.g. `distilbert-base-uncased`), add a multi-label classification head
3. **Training** — fine-tune with binary cross-entropy loss; track per-label F1
4. **Evaluation** — report micro/macro F1, precision, recall; analyse failure cases
5. **Inference** — run the trained model on new email text with a configurable threshold

## Setup

```bash
uv sync
```
