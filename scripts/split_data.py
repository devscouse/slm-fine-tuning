"""Split validated JSONL file(s) into train / val / test sets (70/15/15)."""

import argparse
import hashlib
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.model_selection import train_test_split


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicate records by (subject, body) hash."""
    seen: set[str] = set()
    unique: list[dict] = []
    for r in records:
        key = hashlib.sha256(
            (r.get("subject", "") + r.get("body", "")).encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Split email JSONL into train/val/test.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/synthetic/emails.jsonl"],
        help="One or more JSONL input files",
    )
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Concatenate records from all input files
    records: list[dict] = []
    for input_path in args.input:
        p = Path(input_path)
        if p.exists():
            records.extend(_read_jsonl(p))
        else:
            print(f"Warning: {input_path} not found, skipping")

    records = _deduplicate(records)
    outdir = Path(args.outdir)

    labels = [r["label"] for r in records]

    train, temp, train_labels, temp_labels = train_test_split(
        records, labels, test_size=0.30, random_state=args.seed, stratify=labels,
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=args.seed, stratify=temp_labels,
    )

    _write_jsonl(train, outdir / "train.jsonl")
    _write_jsonl(val, outdir / "val.jsonl")
    _write_jsonl(test, outdir / "test.jsonl")

    print(f"Total (deduped): {len(records)}")
    print(f"Train          : {len(train)}")
    print(f"Val            : {len(val)}")
    print(f"Test           : {len(test)}")


if __name__ == "__main__":
    main()
