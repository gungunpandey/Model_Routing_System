"""
Extract questions from a JSONL of QA pairs and auto-label difficulty (simple/medium/complex).

Input JSONL format (one object per line):
  {"question": "...", "answer": "...", ...}

Usage:
  python data/label_questions.py --input data/qa_final_pairs.jsonl --output data/labeled.jsonl --val_ratio 0.1
  # With validation split and minimum length filter:
  python data/label_questions.py --input data/qa_final_pairs.jsonl --output data/labeled_train.jsonl --val_output data/labeled_val.jsonl --val_ratio 0.1 --min_len 25

Notes:
  - Labels are heuristic; review and adjust rules/thresholds as needed.
  - The script shuffles and can emit train/val splits if --val_ratio > 0.
"""

import argparse
import json
import random
from pathlib import Path


def is_questionish(text: str, min_len: int) -> bool:
    """Basic check to filter out titles/headlines that are not real questions."""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < min_len:
        return False
    lower = stripped.lower()
    if lower.startswith(("http://", "https://")):
        return False
    has_qmark = "?" in stripped
    question_words = (
        "what", "why", "how", "when", "where", "which", "who",
        "can", "should", "do", "does", "did", "is", "are",
        "could", "would", "will", "may", "shall",
    )
    starts_qword = lower.startswith(question_words)
    # Titles with separators and no question mark/word are likely not questions.
    looks_like_title = (" - " in stripped or " | " in stripped) and not has_qmark and not starts_qword
    if looks_like_title:
        return False
    # Allow questions without a '?' as long as they aren't titles and meet length.
    return True


def classify_question(text: str) -> str:
    """Heuristic difficulty classifier based on length and keywords."""
    t = text.lower()
    length = len(t)

    complex_keywords = [
        "design", "optimize", "stability", "compensation", "thermal", "efficiency",
        "isolation", "emi", "ringing", "overshoot", "layout", "pcb", "buck", "boost",
        "adc", "dac", "filter", "pll", "rf", "impedance", "transient", "protection",
        "esd", "safety", "diagnose", "troubleshoot", "trade-off", "analysis",
    ]
    medium_keywords = [
        "calculate", "compute", "derive", "select", "choose", "configure", "interface",
        "debounce", "level shift", "gain", "cutoff", "frequency", "current", "voltage",
        "resistor", "capacitor", "inductor", "op-amp", "mosfet", "bjt",
        "spi", "i2c", "uart", "pwm",
    ]
    troubleshoot_keywords = [
        "troubleshoot", "debug", "fix", "failure", "error", "issue", "bug",
        "latency", "delay", "dropped", "unstable", "reset", "fault", "crash",
        "jitter", "noise", "glitch", "disconnect", "overrun", "underrun",
        "usb", "adapter", "driver",
    ]

    complex_hits = sum(1 for k in complex_keywords if k in t)
    medium_hits = sum(1 for k in medium_keywords if k in t)
    troubleshoot_hits = sum(1 for k in troubleshoot_keywords if k in t)

    has_code = "```" in text or "int " in text or "void " in text or "#include" in text

    # Troubleshooting is at least medium; longer or with complex terms -> complex.
    if troubleshoot_hits > 0:
        if length > 220 or complex_hits > 0:
            return "complex"
        return "medium"

    # Length-based heuristics (characters, rough proxy)
    if length > 450 or complex_hits >= 2:
        return "complex"
    if has_code and (medium_hits > 0 or complex_hits > 0):
        return "medium"
    if length < 140 and medium_hits == 0 and complex_hits == 0:
        return "simple"
    if 140 <= length <= 320 and (medium_hits > 0) and complex_hits == 0:
        return "medium"
    if medium_hits >= 2 and complex_hits == 0:
        return "medium"
    # Fallback
    if complex_hits > 0:
        return "complex"
    return "medium"


def load_questions(path: Path, min_len: int):
    kept, skipped = 0, 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("question")
            if q and is_questionish(q, min_len):
                kept += 1
                yield q.strip()
            else:
                skipped += 1
    print(f"Filtered questions: kept={kept}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source QA JSONL")
    parser.add_argument("--output", required=True, help="Path to labeled JSONL")
    parser.add_argument("--val_output", help="Optional path for validation split JSONL")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Fraction for validation split (0-0.5)")
    parser.add_argument("--min_len", type=int, default=20, help="Minimum question length to keep")
    args = parser.parse_args()

    questions = list(load_questions(Path(args.input), args.min_len))
    random.shuffle(questions)

    labeled = [{"text": q, "label": classify_question(q)} for q in questions]

    val_size = 0
    if args.val_output and 0 < args.val_ratio < 0.5:
        val_size = int(len(labeled) * args.val_ratio)

    val_set = labeled[:val_size]
    train_set = labeled[val_size:]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.val_output and val_set:
        val_path = Path(args.val_output)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        with val_path.open("w", encoding="utf-8") as f:
            for item in val_set:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_set)} train rows to {args.output}")
    if args.val_output and val_set:
        print(f"Wrote {len(val_set)} val rows to {args.val_output}")


if __name__ == "__main__":
    main()

