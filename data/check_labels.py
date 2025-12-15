"""
Quick sanity checks for labeled data (JSONL with fields: text, label).

Usage:
  python data/check_labels.py --input data/labeled_train.jsonl

What it reports:
  - Total rows
  - Label counts and percentages
  - Average/median text length per label (in characters)
  - A few sample rows per label
"""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to labeled JSONL (fields: text, label)")
    parser.add_argument("--samples", type=int, default=3, help="Sample rows per label to display")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if not rows:
        print("No rows loaded.")
        return

    total = len(rows)
    labels = [r.get("label") for r in rows if "label" in r]
    counts = Counter(labels)

    print(f"Total rows: {total}")
    for lbl, cnt in counts.most_common():
        pct = 100.0 * cnt / total
        print(f"  {lbl}: {cnt} ({pct:.1f}%)")

    # Length stats per label
    lengths = defaultdict(list)
    for r in rows:
        lbl = r.get("label")
        txt = r.get("text", "")
        lengths[lbl].append(len(txt))

    print("\nLength stats (chars) per label:")
    for lbl, lens in lengths.items():
        avg = sum(lens) / len(lens)
        med = statistics.median(lens)
        print(f"  {lbl}: avg={avg:.1f}, median={med:.1f}, min={min(lens)}, max={max(lens)}")

    # Samples per label
    print("\nSamples:")
    for lbl in counts:
        subset = [r for r in rows if r.get("label") == lbl]
        take = min(args.samples, len(subset))
        print(f"\nLabel: {lbl} (showing {take})")
        for r in random.sample(subset, take):
            txt = r.get("text", "").replace("\n", " ")
            print(f"- {txt[:200]}{'...' if len(txt) > 200 else ''}")


if __name__ == "__main__":
    main()

