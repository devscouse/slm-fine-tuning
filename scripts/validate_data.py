"""Validate and deduplicate a JSONL file of labelled emails."""

import argparse
import hashlib
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_triage.labels import LABEL_NAMES


def _hash(record: dict) -> str:
    key = (record["subject"].strip().lower(), record["body"].strip().lower())
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def validate(record: dict) -> str | None:
    """Return an error message if invalid, else None."""
    if not isinstance(record.get("subject"), str) or not record["subject"].strip():
        return "missing/empty subject"
    if not isinstance(record.get("body"), str) or not record["body"].strip():
        return "missing/empty body"
    labels = record.get("labels")
    if not isinstance(labels, list) or len(labels) == 0:
        return "labels must be a non-empty list"
    unknown = [l for l in labels if l not in LABEL_NAMES]
    if unknown:
        return f"unknown labels: {unknown}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and deduplicate email JSONL.")
    parser.add_argument("--input", default="data/synthetic/emails.jsonl")
    parser.add_argument("--output", default=None, help="Defaults to --input (in-place)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    invalid = 0
    duplicates = 0
    seen: set[str] = set()
    clean: list[dict] = []

    for record in records:
        err = validate(record)
        if err:
            invalid += 1
            continue
        h = _hash(record)
        if h in seen:
            duplicates += 1
            continue
        seen.add(h)
        clean.append(record)

    kept = len(clean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in clean:
            f.write(json.dumps(record) + "\n")

    print(f"Total records  : {total}")
    print(f"Invalid        : {invalid}")
    print(f"Duplicates     : {duplicates}")
    print(f"Kept           : {kept}")


if __name__ == "__main__":
    main()
