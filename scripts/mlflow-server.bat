@echo off
REM Start the MLflow tracking server (UI at http://localhost:5000)
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///outputs/mlflow.db --default-artifact-root ./mlruns
