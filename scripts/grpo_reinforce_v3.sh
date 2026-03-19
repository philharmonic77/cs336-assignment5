#!/usr/bin/env bash
set -euo pipefail

python cs336_alignment/grpo.py \
  REINFORCE_v3 \
  --learning-rate 3e-5 \
  --no-normalize-by-std