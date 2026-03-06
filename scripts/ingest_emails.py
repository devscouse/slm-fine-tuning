"""Batch-parse .eml/.msg files from a directory into JSONL.

Usage:
    python scripts/ingest_emails.py [--input-dir data/raw] [--output data/raw/emails.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_triage.data.email_parser import parse_directory


def _read_source_files(path: Path) -> set[str]:
    """Read all source_file values from an existing JSONL file."""
    sources: set[str] = set()
    if not path.exists():
        return sources
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if "source_file" in rec:
                    sources.add(rec["source_file"])
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest .eml/.msg files into JSONL.")
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing .eml/.msg files",
    )
    parser.add_argument(
        "--output",
        default="data/raw/emails.jsonl",
        help="Output JSONL file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # Collect already-ingested source files
    existing = _read_source_files(output_path)

    # Parse all email files
    parsed = parse_directory(input_dir)

    # Filter out duplicates and errors
    new_records = []
    errors = []
    skipped = 0
    for rec in parsed:
        if "error" in rec:
            errors.append(rec)
        elif rec["source_file"] in existing:
            skipped += 1
        else:
            new_records.append(rec)

    # Append to output
    if new_records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Scanned        : {len(parsed)} files")
    print(f"New records     : {len(new_records)}")
    print(f"Duplicates      : {skipped}")
    print(f"Errors          : {len(errors)}")

    for err in errors:
        print(f"  - {err['source_file']}: {err['error']}")


if __name__ == "__main__":
    main()
