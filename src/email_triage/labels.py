"""
Label taxonomy for the email triage multi-class classification task.

Every email is assigned exactly one of four mutually exclusive classes:
Attention, Notice, Ignore, Security.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassDef:
    name: str
    description: str


CLASSES: list[ClassDef] = [
    ClassDef(
        name="attention",
        description="Requires the recipient to take a meaningful action — something will be missed or delayed otherwise.",
    ),
    ClassDef(
        name="notice",
        description="Contains useful information but does not require any action. Reading is enough.",
    ),
    ClassDef(
        name="ignore",
        description="Entirely unimportant for day-to-day activity; exists mainly as a searchable record.",
    ),
    ClassDef(
        name="security",
        description="Related to account security or identity verification (MFA codes, password resets, alerts).",
    ),
]

CLASS_NAMES: list[str] = [c.name for c in CLASSES]
NUM_CLASSES: int = len(CLASS_NAMES)

# Backward-compat aliases
LABEL_NAMES = CLASS_NAMES
NUM_LABELS = NUM_CLASSES

_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def label_to_index(label: str) -> int:
    """Convert a class name string to its integer index."""
    try:
        return _NAME_TO_INDEX[label]
    except KeyError:
        raise ValueError(f"Unknown class label {label!r}. Valid: {CLASS_NAMES}") from None


def index_to_label(index: int) -> str:
    """Convert an integer index back to its class name string."""
    if not 0 <= index < NUM_CLASSES:
        raise ValueError(f"Index {index} out of range [0, {NUM_CLASSES})")
    return CLASS_NAMES[index]
