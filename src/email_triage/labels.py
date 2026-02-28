"""
Label taxonomy for the email triage multi-label classification task.

An email can carry any combination of these labels simultaneously.
For example, a time-sensitive purchase confirmation might be:
    [URGENT, ACTION_REQUIRED, ORDER_CONFIRMATION]
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    name: str
    description: str


LABELS: list[Label] = [
    Label(
        name="urgent",
        description="Time-sensitive; needs attention today or risk of a missed deadline / escalation.",
    ),
    Label(
        name="needs_reply",
        description="The sender explicitly expects or is waiting for a response.",
    ),
    Label(
        name="action_required",
        description="The recipient must take a concrete action (approve, sign, submit, fix, etc.).",
    ),
    Label(
        name="order_confirmation",
        description="Confirms a purchase, shipment, or subscription sign-up.",
    ),
    Label(
        name="alerts",
        description="Automated system or service alert (security notice, outage, threshold breach).",
    ),
    Label(
        name="calendar_event",
        description="Meeting invite, scheduling request, RSVP, or calendar update.",
    ),
    Label(
        name="newsletters",
        description="Newsletter, marketing digest, promotional email, or mailing-list post.",
    ),
]

# Ordered list of label names — used as the fixed output vector for the model.
LABEL_NAMES: list[str] = [label.name for label in LABELS]
NUM_LABELS: int = len(LABEL_NAMES)


def label_vector(active_labels: list[str]) -> list[int]:
    """Convert a list of active label names into a binary vector."""
    return [1 if name in active_labels else 0 for name in LABEL_NAMES]


def vector_to_labels(vector: list[int], threshold: float = 0.5) -> list[str]:
    """Convert a binary (or probability) vector back into label names."""
    return [name for name, val in zip(LABEL_NAMES, vector) if val >= threshold]
