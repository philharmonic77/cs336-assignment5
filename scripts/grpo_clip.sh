#!/usr/bin/env bash
set -euo pipefail

python cs336_alignment/grpo.py \
  grpo_clip \
  --loss-type grpo_clip \
  --train-batch-size 128 \
  --gradient-accumulation-steps 128 \
  --epochs-per-rollout-batch 2 \
  --learning-rate 5e-6
