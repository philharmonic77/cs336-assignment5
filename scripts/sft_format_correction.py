#!/usr/bin/env python3
"""
Convert SFT data to corrected JSONL format.

Logic: if response contains both <answer> and </think>,
move the first </think> to immediately before the first <answer>
with exactly one space between them.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def iter_records_from_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from e


def correct_response(response: str) -> str:
    if "<answer>" in response and "</think>" in response:
        response = response.replace("</think>", "", 1)
        response = response.replace("<answer>", "</think> <answer>", 1)
    return response


def is_format_correct(response: str) -> bool:
    return "</think> <answer>" in response


def process_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rec in records:
        if "response" in rec and isinstance(rec["response"], str):
            rec = dict(rec)
            rec["response"] = correct_response(rec["response"])
            rec["format_correct"] = is_format_correct(rec["response"])
        if rec.get("format_correct") is True:
            rec["metrics"] = r1_zero_reward_fn(
                rec.get("response", ""),
                rec.get("answer"),
                fast=True,
            )
            out.append(rec)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Correct SFT response format and output JSONL.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    in_path = base_dir / "data/math/sft.jsonl"
    out_path = in_path.parent / "sft_format_correction.jsonl"

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1

    records = iter_records_from_jsonl(in_path)

    processed = process_records(records)
    correct_count = len(processed)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in processed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(processed)} records to {out_path}")
    print(f"Format-correct records: {correct_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
