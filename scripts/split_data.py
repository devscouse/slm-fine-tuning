"""Split a validated JSONL into train / val / test sets (70/15/15)."""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Split email JSONL into train/val/test.")
    parser.add_argument("--input", default="data/synthetic/emails.jsonl")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _read_jsonl(Path(args.input))
    outdir = Path(args.outdir)

    train, temp = train_test_split(records, test_size=0.30, random_state=args.seed)
    val, test = train_test_split(temp, test_size=0.50, random_state=args.seed)

    _write_jsonl(train, outdir / "train.jsonl")
    _write_jsonl(val, outdir / "val.jsonl")
    _write_jsonl(test, outdir / "test.jsonl")

    print(f"Total          : {len(records)}")
    print(f"Train          : {len(train)}")
    print(f"Val            : {len(val)}")
    print(f"Test           : {len(test)}")


if __name__ == "__main__":
    main()
