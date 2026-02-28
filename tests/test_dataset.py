"""Unit tests for EmailDataset."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from email_triage.data import EmailDataset
from email_triage.labels import NUM_LABELS, LABEL_NAMES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MAX_LENGTH = 32

SAMPLE_RECORDS = [
    {
        "subject": "Urgent invoice due",
        "body": "Please pay by end of day.",
        "labels": ["urgent", "action_required"],
    },
    {
        "subject": "Weekly newsletter",
        "body": "Here are the top stories this week.",
        "labels": ["newsletters"],
    },
    {
        "subject": "Order shipped",
        "body": "Your order #1234 has shipped.",
        "labels": ["order_confirmation"],
    },
]


def _mock_tokenizer(max_length: int = MAX_LENGTH) -> MagicMock:
    """Returns a MagicMock that behaves like a HuggingFace tokenizer."""
    tok = MagicMock()

    def side_effect(text, padding, truncation, max_length, return_tensors):
        return {
            "input_ids": torch.zeros(1, max_length, dtype=torch.long),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }

    tok.side_effect = side_effect
    return tok


@pytest.fixture()
def jsonl_file(tmp_path: Path) -> Path:
    path = tmp_path / "emails.jsonl"
    with path.open("w") as f:
        for record in SAMPLE_RECORDS:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture()
def dataset(jsonl_file: Path) -> EmailDataset:
    return EmailDataset(jsonl_file, _mock_tokenizer(MAX_LENGTH), max_length=MAX_LENGTH)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_len(dataset: EmailDataset) -> None:
    assert len(dataset) == len(SAMPLE_RECORDS)


def test_item_keys(dataset: EmailDataset) -> None:
    item = dataset[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}


def test_label_vector_shape(dataset: EmailDataset) -> None:
    item = dataset[0]
    assert item["labels"].shape == (NUM_LABELS,)
    assert item["labels"].dtype == torch.float32


def test_label_values(dataset: EmailDataset) -> None:
    # First record: urgent + action_required
    item = dataset[0]
    labels_tensor = item["labels"]
    expected = torch.tensor(
        [1 if n in ("urgent", "action_required") else 0 for n in LABEL_NAMES],
        dtype=torch.float,
    )
    assert torch.equal(labels_tensor, expected)


def test_input_ids_shape(dataset: EmailDataset) -> None:
    item = dataset[0]
    assert item["input_ids"].shape == (MAX_LENGTH,)
