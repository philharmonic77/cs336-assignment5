#!/usr/bin/env bash
set -euo pipefail

SIZES=(128 256 512 1024 2048 4096 8192 15000)

for size in "${SIZES[@]}"; do
  out_dir="outputs/sft/train_sample_size_${size}"
  mkdir -p "${out_dir}"
  echo "=== Running train-sample-size=${size} ==="
  python -u cs336_alignment/sft.py \
    --train-sample-size "${size}" \
    --output-dir "${out_dir}"
done
