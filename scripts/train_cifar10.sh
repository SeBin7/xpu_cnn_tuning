#!/usr/bin/env bash
set -euo pipefail
python -u src/train.py --config ./configs/cifar10.yaml
