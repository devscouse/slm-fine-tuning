"""
Model module for multi-class email triage classification.

Backbone is model-agnostic: any HuggingFace encoder can be selected by name.
The classification head is a single linear layer over a pooled representation.
Training uses CrossEntropyLoss (raw logits); inference uses predict_proba (softmax).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from email_triage.labels import NUM_CLASSES


class EmailTriageClassifier(nn.Module):
    """
    Encoder backbone + classification head for multi-class email triage.

    Args:
        model_name: HuggingFace model identifier (e.g. "distilbert-base-uncased").
        num_classes: Number of output classes. Defaults to NUM_CLASSES (4).
        dropout:     Dropout probability applied before the linear head.
        pooling:     "cls" uses the first token; "mean" averages non-padding tokens.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        pooling: Literal["cls", "mean"] = "cls",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling

        config = AutoConfig.from_pretrained(model_name)
        hidden_size: int = config.hidden_size

        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pool(
        self,
        hidden_states: torch.Tensor,   # (B, L, H)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, H)
        if self.pooling == "cls":
            return hidden_states[:, 0, :]

        # Mean pooling — ignore padding positions
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)           # (B, H)
        return summed / mask.sum(dim=1).clamp(min=1e-9)      # (B, H)

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, num_classes) — raw logits
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))

    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, num_classes) — probabilities summing to 1
        return torch.softmax(self.forward(input_ids, attention_mask), dim=-1)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_classifier(model_name: str, **kwargs) -> EmailTriageClassifier:
    """Construct an EmailTriageClassifier for the given HuggingFace model name."""
    return EmailTriageClassifier(model_name, **kwargs)


def build_tokenizer(model_name: str):
    """
    Return a fast tokenizer for *model_name*.

    Import is deferred so the classifier module can be imported without
    incurring tokenizer initialisation overhead when only the model is needed.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
