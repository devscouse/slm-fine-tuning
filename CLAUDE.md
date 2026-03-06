# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Email triage classifier: fine-tune a small language model (DistilBERT) for **multi-class classification** of emails into 4 mutually exclusive classes: **Attention**, **Notice**, **Ignore**, **Security**. See `REQUIREMENTS.md` for class definitions and examples.

**Real email ingestion**: upload `.eml`/`.msg` files via the Streamlit dashboard (`app/labeling.py`) or batch-parse with `scripts/ingest_emails.py`. All real emails (labeled and unlabeled) are stored in `data/raw/emails.jsonl`.

## Commands

```bash
uv sync                          # Install all dependencies
pytest                           # Run all tests
pytest tests/test_training.py    # Run a single test file
pytest -k "test_name"            # Run a single test by name
ruff check .                     # Lint
ruff format .                    # Auto-format
```

### Data pipeline scripts

```bash
python scripts/generate_synthetic_data.py --count 100   # Requires GOOGLE_API_KEY env var
python scripts/split_data.py                             # Split into train/val/test (70/15/15)
python scripts/split_data.py --input data/synthetic/emails.jsonl data/raw/emails.jsonl
python scripts/train.py                                  # Train model (CLI args for hyperparams)
python scripts/ingest_emails.py                          # Parse .eml/.msg from data/raw/
streamlit run app/labeling.py                            # Streamlit dashboard for labeling
```

## Architecture

```
app/
└── labeling.py            # Streamlit dashboard (upload, label, browse, stats/export)

src/email_triage/
├── labels.py              # Label taxonomy, class-to-index mappings
├── data/
│   ├── dataset.py         # PyTorch Dataset — loads JSONL, tokenizes
│   ├── email_parser.py    # Parse .eml/.msg files into subject + body dicts
│   └── llm.py             # LLM provider abstraction (Gemini) for synthetic data
├── model/
│   └── classifier.py      # EmailTriageClassifier: HuggingFace backbone + classification head
├── training/
│   ├── trainer.py         # Training loop, AdamW, checkpointing, MLflow logging
│   └── metrics.py         # Evaluation metrics (F1, precision, recall via sklearn)
└── inference/             # Not yet implemented
```

**Data format** (JSONL): each record has `subject`, `body`, and a class label field.

**Model**: encoder-only transformer (default `distilbert-base-uncased`) with a linear classification head. Supports CLS-token or mean pooling.

**Training**: AdamW optimizer, gradient clipping, best-model checkpointing by validation macro-F1, MLflow experiment tracking (`outputs/mlflow.db`).
