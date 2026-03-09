#!/usr/bin/env bash
set -euo pipefail

SIZE=4096
ONLY_USE_CORRECT_VALUES=(false true)

for only_use_correct in "${ONLY_USE_CORRECT_VALUES[@]}"; do
  suffix="only_correct_${only_use_correct}"
  out_dir="outputs/sft/train_sample_size_${SIZE}_${suffix}"
  mkdir -p "${out_dir}"
  echo "=== Running train-sample-size=${SIZE} only_use_correct=${only_use_correct} ==="

  cmd=(python -u cs336_alignment/sft.py --train-sample-size "${SIZE}" --output-dir "${out_dir}")
  if [[ "${only_use_correct}" == "true" ]]; then
    cmd+=(--only-use-correct)
  fi
  "${cmd[@]}"
done
