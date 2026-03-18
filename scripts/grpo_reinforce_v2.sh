#!/usr/bin/env bash
set -euo pipefail

python cs336_alignment/grpo.py \
  REINFORCE_v2 \
  --learning-rate 4e-5 \
  --gradient-accumulation-steps 64 \
  --no-normalize-by-std
