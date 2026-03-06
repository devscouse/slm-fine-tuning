# SLM Fine-Tuning — Email Triage: Task List

## Data
- [ ] Generate full synthetic dataset (target ≥ 750 emails; currently 0 in new format)
- [x] Validate & de-duplicate JSONL records (`scripts/validate_data.py`)
- [x] Add core ML dependencies to pyproject.toml (torch, transformers, scikit-learn, numpy, pandas, pytest)
- [x] Implement PyTorch `EmailDataset` class (`src/email_triage/data/dataset.py`)
- [x] Build train / val / test splits (70 / 15 / 15) → `data/processed/` (`scripts/split_data.py`)

## Model
- [x] Define model-agnostic backbone + multi-class classification head (`src/email_triage/model/classifier.py`)
- [x] Unit-test model forward pass (output shape, softmax range, pooling correctness)
- [ ] Wire `index_to_label` helper into inference layer (argmax-based)

## Training & Evaluation (MLflow)
- [ ] Set up MLflow experiment and autolog run tracking (`outputs/mlflow.db`)
- [ ] Implement training loop: AdamW, cross-entropy loss, gradient clipping
- [ ] Per-epoch validation: per-class F1, precision, recall logged to MLflow
- [ ] Implement accuracy / macro F1 final evaluation and save report
- [ ] Checkpoint best model by validation macro-F1 (`models/best.pt`)
- [ ] Hyperparameter sweep (LR, batch size, epochs) via MLflow param logging

## Inference
- [ ] Implement argmax-based multi-class predictor (`src/email_triage/inference/`)
- [ ] CLI entry-point: accept subject + body, print predicted class + confidence
- [ ] Unit-test predictor (known inputs → expected class)

## Deployment / Integration
- [ ] Wrap predictor in FastAPI service with `/predict` endpoint
- [ ] Define Pydantic request/response schema
- [ ] Containerise with Docker (multi-stage build, non-root user)
- [ ] Integration smoke-test against running container

## Monitoring & Reporting (Streamlit)
- [ ] Streamlit app skeleton with sidebar navigation
- [ ] Dataset page: class distribution bar chart, example browser
- [ ] Model performance page: per-class F1/precision/recall table, confusion matrix
- [ ] Live inference demo: text inputs → predicted class with confidence scores
- [ ] MLflow metrics page: loss/F1 curves pulled from `outputs/mlflow.db`
