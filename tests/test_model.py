"""Unit tests for EmailTriageClassifier — no network access required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from email_triage.labels import NUM_CLASSES
from email_triage.model import EmailTriageClassifier, build_classifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 64   # deliberately not 768 to catch hardcoded-768 bugs
BATCH_SIZE = 2
SEQ_LEN = 32
MODEL_NAME = "mock-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.hidden_size = HIDDEN_SIZE
    return cfg


def _make_backbone(hidden_size: int = HIDDEN_SIZE) -> MagicMock:
    backbone = MagicMock()
    output = MagicMock()
    output.last_hidden_state = torch.randn(BATCH_SIZE, SEQ_LEN, hidden_size)
    backbone.return_value = output
    return backbone


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    # Last 8 positions are padding (mask=0) to exercise mean-pooling branch
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    attention_mask[:, -8:] = 0
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def classifier() -> EmailTriageClassifier:
    with (
        patch("email_triage.model.classifier.AutoConfig") as mock_cfg,
        patch("email_triage.model.classifier.AutoModel") as mock_model,
    ):
        mock_cfg.from_pretrained.return_value = _make_config()
        mock_model.from_pretrained.return_value = _make_backbone()
        yield EmailTriageClassifier(MODEL_NAME)


@pytest.fixture()
def mean_classifier() -> EmailTriageClassifier:
    with (
        patch("email_triage.model.classifier.AutoConfig") as mock_cfg,
        patch("email_triage.model.classifier.AutoModel") as mock_model,
    ):
        mock_cfg.from_pretrained.return_value = _make_config()
        mock_model.from_pretrained.return_value = _make_backbone()
        yield EmailTriageClassifier(MODEL_NAME, pooling="mean")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_output_shape(classifier: EmailTriageClassifier) -> None:
    input_ids, attention_mask = _make_inputs()
    logits = classifier(input_ids, attention_mask)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert logits.dtype == torch.float32


def test_forward_returns_raw_logits(classifier: EmailTriageClassifier) -> None:
    input_ids, attention_mask = _make_inputs()
    logits = classifier(input_ids, attention_mask)
    # Raw logits are NOT constrained to [0, 1]; at least some should exceed bounds
    # with random weights. We check they are not all sigmoid-squashed.
    assert not (logits.min() >= 0.0 and logits.max() <= 1.0), (
        "forward() should return raw logits, not probabilities"
    )


def test_predict_proba_range(classifier: EmailTriageClassifier) -> None:
    input_ids, attention_mask = _make_inputs()
    probs = classifier.predict_proba(input_ids, attention_mask)
    assert probs.shape == (BATCH_SIZE, NUM_CLASSES)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_proba_sums_to_one(classifier: EmailTriageClassifier) -> None:
    input_ids, attention_mask = _make_inputs()
    probs = classifier.predict_proba(input_ids, attention_mask)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=1e-5)


def test_predict_proba_no_grad(classifier: EmailTriageClassifier) -> None:
    input_ids, attention_mask = _make_inputs()
    probs = classifier.predict_proba(input_ids, attention_mask)
    assert not probs.requires_grad


def test_cls_pooling_uses_first_token(classifier: EmailTriageClassifier) -> None:
    hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    result = classifier._pool(hidden, mask)
    assert torch.equal(result, hidden[:, 0, :])


def test_mean_pooling_ignores_padding(mean_classifier: EmailTriageClassifier) -> None:
    # Build hidden states where the padded positions are extreme values.
    # If padding is not masked, the mean will be polluted.
    hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    mask[:, -8:] = 0  # last 8 tokens are padding

    # Poison the padding positions with a large constant
    hidden_poisoned = hidden.clone()
    hidden_poisoned[:, -8:, :] = 1e6

    result_clean = mean_classifier._pool(hidden, mask)
    result_poisoned = mean_classifier._pool(hidden_poisoned, mask)

    # If padding is correctly masked, both results must be identical
    assert torch.allclose(result_clean, result_poisoned, atol=1e-5), (
        "Mean pooling must ignore positions where attention_mask == 0"
    )


def test_build_classifier_returns_correct_type() -> None:
    with (
        patch("email_triage.model.classifier.AutoConfig") as mock_cfg,
        patch("email_triage.model.classifier.AutoModel") as mock_model,
    ):
        mock_cfg.from_pretrained.return_value = _make_config()
        mock_model.from_pretrained.return_value = _make_backbone()
        model = build_classifier(MODEL_NAME)
    assert isinstance(model, EmailTriageClassifier)


def test_model_name_stored(classifier: EmailTriageClassifier) -> None:
    assert classifier.model_name == MODEL_NAME
