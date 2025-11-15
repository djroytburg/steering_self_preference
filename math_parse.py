#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

from math_parser import extract_answer, strip_answer_string, math_equal


def evaluate_file(input_path: str, output_path: str | None = None):
    total = 0
    correct = 0

    per_subject_total = defaultdict(int)
    per_subject_correct = defaultdict(int)

    # Open output file if we want to write the augmented dataset
    f_out = open(output_path, "w", encoding="utf-8") if output_path else None

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            subject = rec.get("subject", "unknown")
            gt_raw = rec.get("solution", "")
            pred_raw = rec.get("response", "") or ""

            # Clean ground truth and prediction
            gt = strip_answer_string(gt_raw)
            pred = extract_answer(pred_raw)

            ok = math_equal(pred, gt)

            total += 1
            per_subject_total[subject] += 1
            if ok:
                correct += 1
                per_subject_correct[subject] += 1

            # ---- add correctness flag to record ----
            rec["answer_correct"] = 1 if ok else 0

            # write augmented record if requested
            if f_out is not None:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if f_out is not None:
        f_out.close()

    return total, correct, per_subject_total, per_subject_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL file to evaluate",
    )
    parser.add_argument(
        "--output",
        help="Path to JSONL file to write augmented records with answer_correct (optional)",
    )
    args = parser.parse_args()

    total, correct, per_sub_tot, per_sub_corr = evaluate_file(
        args.input,
        args.output,
    )

    print(f"Total examples:   {total}")
    print(f"Total correct:    {correct}")
    if total > 0:
        print(f"Overall accuracy: {correct / total:.4f}")
    print()

    print("Per-subject accuracy:")
    for subject in sorted(per_sub_tot.keys()):
        t = per_sub_tot[subject]
        c = per_sub_corr[subject]
        acc = c / t if t > 0 else 0.0
        print(f"  {subject:25s} {c:5d} / {t:5d}  = {acc:.4f}")


if __name__ == "__main__":
    main()

