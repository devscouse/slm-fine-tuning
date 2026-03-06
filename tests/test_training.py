"""Tests for training metrics and config."""

import numpy as np
import pytest

from email_triage.labels import CLASS_NAMES
from email_triage.training.metrics import EvalResult, compute_metrics
from email_triage.training.trainer import TrainConfig


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])

        result = compute_metrics(y_true, y_pred)

        assert isinstance(result, EvalResult)
        assert result.macro_f1 == pytest.approx(1.0)
        assert result.accuracy == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([1, 0, 3, 2])

        result = compute_metrics(y_true, y_pred)

        assert result.macro_f1 == pytest.approx(0.0)
        assert result.accuracy == pytest.approx(0.0)

    def test_per_class_keys(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])

        result = compute_metrics(y_true, y_pred)

        assert set(result.per_class.keys()) == set(CLASS_NAMES)
        for name in CLASS_NAMES:
            assert "f1" in result.per_class[name]
            assert "precision" in result.per_class[name]
            assert "recall" in result.per_class[name]

    def test_report_text_is_string(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = compute_metrics(y_true, y_pred)

        assert isinstance(result.report_text, str)
        assert len(result.report_text) > 0


class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig()

        assert config.model_name == "distilbert-base-uncased"
        assert config.lr == 2e-5
        assert config.epochs == 10
        assert config.batch_size == 16
        assert config.max_grad_norm == 1.0
        assert config.data_dir == "data/processed"
        assert config.checkpoint_path == "models/best.pt"
        assert config.experiment_name == "email-triage"

    def test_override(self):
        config = TrainConfig(lr=1e-4, epochs=5)

        assert config.lr == 1e-4
        assert config.epochs == 5
