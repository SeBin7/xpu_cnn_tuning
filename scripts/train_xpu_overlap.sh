#!/usr/bin/env bash
set -euo pipefail

python -u src/train_xpu_overlap.py \
  --config configs/cifar10_xpu_fused.yaml \
  --fused auto \
  --ipex off \
  --channels-last off \
  --num-workers 4 \
  --prefetch-factor 4 \
  --pin-memory on \
  --persistent-workers on
