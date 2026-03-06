"""
Generate synthetic labelled emails for the email-triage fine-tuning dataset.

Usage:
    uv run python scripts/generate_synthetic_data.py [--count 750] [--batch 10] [--provider gemini] [--out data/synthetic/emails.jsonl]

Environment variables:
    GOOGLE_API_KEY   — required when using the Gemini provider

The script cycles through 4 class-specific scenarios to ensure balanced coverage,
then generates `--batch` emails per API call until `--count` examples have been collected.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Make the src package importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_triage.data.llm import get_provider
from email_triage.labels import CLASS_NAMES, CLASSES


# ---------------------------------------------------------------------------
# Scenario definitions
# Each scenario specifies which class the generated emails should belong to.
# ---------------------------------------------------------------------------

SCENARIOS: list[dict] = [
    {"label": "attention"},
    {"label": "notice"},
    {"label": "ignore"},
    {"label": "security"},
]


def _class_descriptions() -> str:
    return "\n".join(f'  "{c.name}": {c.description}' for c in CLASSES)


def _build_prompt(target_class: str, batch_size: int) -> str:
    desc_str = _class_descriptions()

    return f"""You are generating a synthetic email dataset for a machine-learning project.

Class taxonomy (4 mutually exclusive classes):
{desc_str}

Task:
Generate exactly {batch_size} realistic-looking emails. Each email MUST belong to
the class: "{target_class}"

Rules:
- Vary the writing style (formal, casual, automated/system-generated).
- Vary the domain/industry (tech, retail, healthcare, finance, HR, etc.).
- Do NOT reuse the same subject across emails in this batch.
- Subjects should be concise (under 15 words).
- Bodies should be 2–6 sentences (realistic email length).
- Return ONLY a JSON array — no explanation, no markdown fences.

JSON format (array of {batch_size} objects):
[
  {{
    "subject": "<email subject>",
    "body": "<email body>",
    "label": "{target_class}"
  }},
  ...
]"""


def _parse_response(text: str, target_class: str) -> list[dict]:
    """Extract and validate the JSON array from the model's response."""
    # Strip optional markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)

    try:
        emails = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}\n---\n{text[:500]}") from exc

    if not isinstance(emails, list):
        raise ValueError(f"Expected a JSON array, got {type(emails).__name__}")

    valid = []
    for item in emails:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        body = str(item.get("body", "")).strip()
        label = item.get("label", "")

        if not subject or not body:
            continue
        # Validate label — force to target class if unrecognised
        if label not in CLASS_NAMES:
            label = target_class

        valid.append({"subject": subject, "body": body, "label": label})

    return valid


def _class_frequencies(paths: list[Path]) -> dict[str, float]:
    """Return per-class frequency (appearances / total records) from JSONL files."""
    counts: Counter = Counter()
    total = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    label = record.get("label", "")
                    if label in CLASS_NAMES:
                        counts[label] += 1
                    total += 1
                except (json.JSONDecodeError, AttributeError):
                    print(f"  [warn] skipping malformed line in {path}")
    if total == 0:
        return {name: 0.0 for name in CLASS_NAMES}
    return {name: counts[name] / total for name in CLASS_NAMES}


def _scenario_weights(freqs: dict[str, float]) -> list[float]:
    """Derive per-scenario sampling weights from per-class frequencies.

    Scenarios whose class is underrepresented receive higher weights.
    Falls back to uniform weights when no data exists or distribution is already balanced.
    """
    # No data at all — use uniform weights
    if all(v == 0.0 for v in freqs.values()):
        return [1.0] * len(SCENARIOS)

    target = sum(freqs.values()) / len(CLASS_NAMES)  # mean observed rate
    deficit = {name: max(0.0, target - freqs[name]) for name in CLASS_NAMES}

    # Already balanced — use uniform weights
    if max(deficit.values()) == 0.0:
        return [1.0] * len(SCENARIOS)

    weights = []
    for scenario in SCENARIOS:
        raw = deficit[scenario["label"]]
        weights.append(max(1.0, raw))  # floor of 1.0: no scenario is fully starved
    return weights


def generate(
    provider_name: str,
    target_count: int,
    batch_size: int,
    out_path: Path,
    existing_paths: list[Path] | None = None,
    max_retries: int = 3,
) -> None:
    provider = get_provider(provider_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _existing = list(existing_paths or [])
    if out_path not in _existing:
        _existing.append(out_path)  # always include out_path if it already exists
    freqs = _class_frequencies(_existing)
    weights = _scenario_weights(freqs)
    batches_since_reweight = 0
    REWEIGHT_EVERY = 15

    collected: list[dict] = []

    mode = "adaptive" if any(p.exists() for p in _existing) else "uniform"
    print(f"Generating {target_count} emails in batches of {batch_size} "
          f"using '{provider_name}' [{mode} scenario weighting]...")

    while len(collected) < target_count:
        [scenario] = random.choices(SCENARIOS, weights=weights, k=1)
        batches_since_reweight += 1
        if batches_since_reweight >= REWEIGHT_EVERY:
            # Recompute weights from in-memory collected data
            counts: Counter = Counter()
            for rec in collected:
                label = rec.get("label", "")
                if label in CLASS_NAMES:
                    counts[label] += 1
            if collected:
                mid_freqs = {name: counts[name] / len(collected) for name in CLASS_NAMES}
                weights = _scenario_weights(mid_freqs)
            batches_since_reweight = 0
        target_class = scenario["label"]
        prompt = _build_prompt(target_class, batch_size)

        for attempt in range(1, max_retries + 1):
            try:
                response = provider.complete(prompt)
                batch = _parse_response(response, target_class)
                break
            except (ValueError, Exception) as exc:
                if attempt == max_retries:
                    print(f"  [warn] Skipping batch after {max_retries} failures: {exc}")
                    batch = []
                else:
                    wait = 2 ** attempt
                    print(f"  [retry {attempt}/{max_retries}] {exc} — waiting {wait}s")
                    time.sleep(wait)

        collected.extend(batch)
        print(f"  {len(collected)}/{target_count} collected (class: {target_class})")

        # Brief pause to stay within free-tier rate limits
        time.sleep(1.0)

    # Trim to exact target and shuffle
    random.shuffle(collected)
    collected = collected[:target_count]

    with out_path.open("a", encoding="utf-8") as f:
        for email in collected:
            f.write(json.dumps(email, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(collected)} emails to {out_path}")

    # --- this run ---
    class_counts: Counter = Counter()
    for email in collected:
        class_counts[email["label"]] += 1
    print(f"\nClass distribution — this run ({len(collected)} emails):")
    for name in CLASS_NAMES:
        count = class_counts[name]
        pct = count / len(collected) * 100
        print(f"  {name:<20} {count:>4}  ({pct:.1f}%)")

    # --- full dataset on disk ---
    full_freqs = _class_frequencies([out_path])
    full_total = sum(1 for _ in out_path.open(encoding="utf-8") if _.strip())
    if full_total > len(collected):          # only worth printing if there's prior data
        print(f"\nClass distribution — full dataset ({full_total} emails):")
        for name in CLASS_NAMES:
            count = round(full_freqs[name] * full_total)
            pct = full_freqs[name] * 100
            print(f"  {name:<20} {count:>4}  ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic labelled email dataset")
    parser.add_argument("--count", type=int, default=750, help="Total emails to generate (default: 750)")
    parser.add_argument("--batch", type=int, default=10, help="Emails per API call (default: 10)")
    parser.add_argument("--provider", default="gemini", help="LLM provider name (default: gemini)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/synthetic/emails.jsonl"),
        help="Output JSONL path (default: data/synthetic/emails.jsonl)",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        nargs="*",
        default=[],
        metavar="FILE",
        help=(
            "Additional JSONL file(s) to read when computing initial class frequencies. "
            "The --out file is always included automatically if it already exists."
        ),
    )
    args = parser.parse_args()

    generate(
        provider_name=args.provider,
        target_count=args.count,
        batch_size=args.batch,
        out_path=args.out,
        existing_paths=args.existing,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    main()
