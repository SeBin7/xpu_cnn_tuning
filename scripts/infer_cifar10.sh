#!/usr/bin/env bash
set -euo pipefail
python -u src/infer.py --config ./configs/cifar10.yaml --ckpt ./outputs/checkpoints/best_cifar10.pt
# Single image example:
# python -u src/infer.py --config ./configs/cifar10.yaml --ckpt ./outputs/checkpoints/best_cifar10.pt --image ./path/to/image.jpg
