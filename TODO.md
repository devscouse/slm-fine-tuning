# SLM Fine-Tuning — Email Triage: Task List

## Data
- [ ] Generate full synthetic dataset (target ≥ 750 emails; currently 160)
- [x] Validate & de-duplicate JSONL records (`scripts/validate_data.py`)
- [x] Add core ML dependencies to pyproject.toml (torch, transformers, scikit-learn, numpy, pandas, pytest)
- [x] Implement PyTorch `EmailDataset` class (`src/email_triage/data/dataset.py`)
- [x] Build train / val / test splits (70 / 15 / 15) → `data/processed/` (`scripts/split_data.py`)

## Model
- [ ] Define DistilBERT backbone + multi-label classification head (`src/email_triage/model/`)
- [ ] Wire `label_vector` / `vector_to_labels` helpers into model output layer
- [ ] Unit-test model forward pass (correct output shape, sigmoid range)

## Training & Evaluation (MLflow)
- [ ] Set up MLflow experiment and autolog run tracking (`outputs/mlruns/`)
- [ ] Implement training loop: AdamW, binary cross-entropy loss, gradient clipping
- [ ] Per-epoch validation: per-label F1, precision, recall logged to MLflow
- [ ] Implement micro / macro F1 final evaluation and save report
- [ ] Checkpoint best model by validation macro-F1 (`models/best.pt`)
- [ ] Hyperparameter sweep (LR, batch size, epochs) via MLflow param logging

## Inference
- [ ] Implement threshold-based multi-label predictor (`src/email_triage/inference/`)
- [ ] Tune per-label decision thresholds on val set; store in config
- [ ] CLI entry-point: accept subject + body, print predicted labels + confidence
- [ ] Unit-test predictor (known inputs → expected label sets)

## Deployment / Integration
- [ ] Wrap predictor in FastAPI service with `/predict` endpoint
- [ ] Define Pydantic request/response schema
- [ ] Containerise with Docker (multi-stage build, non-root user)
- [ ] Integration smoke-test against running container

## Monitoring & Reporting (Streamlit)
- [ ] Streamlit app skeleton with sidebar navigation
- [ ] Dataset page: label distribution bar chart, example browser
- [ ] Model performance page: per-label F1/precision/recall table, confusion matrix
- [ ] Live inference demo: text inputs → predicted labels with confidence scores
- [ ] MLflow metrics page: loss/F1 curves pulled from `outputs/mlruns/`
