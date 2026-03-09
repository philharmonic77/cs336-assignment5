#!/usr/bin/env python3
import argparse
import json
import os
from contextlib import redirect_stderr
from pathlib import Path

from cs336_alignment.drgrpo_grader import extract_answer, grade


def reward_ignore_format_fn(response, answer, fast=True):
    if "<answer>" in response:
        model_answer = response.split("<answer>")[-1].replace("</answer>", "").strip()
        if model_answer and _grade_like_r1_zero(model_answer, answer, fast):
            return True

    model_answer = extract_answer_from_answer_command(response)
    if model_answer and _grade_like_r1_zero(model_answer, answer, fast):
        return True

    model_answer = extract_answer(response)
    if model_answer is None:
        return False
    return _grade_like_r1_zero(model_answer, answer, fast)


def extract_answer_from_answer_command(text: str):
    marker = "\\answer"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None
    i = brace_start
    depth = 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : i].strip()
        i += 1
    return None


def _grade_model_answer(model_answer, answer, fast=True):
    if isinstance(answer, (int, float)):
        answer = str(answer)

    if isinstance(answer, str):
        return grade(model_answer, answer, fast=fast)

    if isinstance(answer, list):
        return any(grade(model_answer, gt, fast=fast) for gt in answer)

    return False


def _grade_like_r1_zero(model_answer, answer, fast=True):
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)
        if model_answer is None:
            return False
    return _grade_model_answer(model_answer, answer, fast)


def main():
    parser = argparse.ArgumentParser(
        description="Check if <answer> or \\boxed{} in response matches the answer field."
    )
    parser.add_argument(
        "--path",
        default="data/math/sft.jsonl",
        help="Path to sft.jsonl (default: data/math/sft.jsonl)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast grading (skip latex equivalence check).",
    )
    parser.add_argument(
        "--print-mismatches",
        action="store_true",
        help="Print indices of mismatches.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    total = 0
    matched = 0
    mismatches = []

    with path.open() as f, open(os.devnull, "w") as devnull:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            response = obj.get("response", "")
            answer = obj.get("answer")
            with redirect_stderr(devnull):
                ok = reward_ignore_format_fn(response, answer, fast=args.fast)
            if ok:
                matched += 1
            else:
                mismatches.append(i)

    print(f"total={total} matched={matched} accuracy={matched / total if total else 0:.4f}")
    if args.print_mismatches:
        print("mismatch_indices=" + ",".join(map(str, mismatches)))


if __name__ == "__main__":
    main()
