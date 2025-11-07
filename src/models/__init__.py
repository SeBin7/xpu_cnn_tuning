from __future__ import annotations
import torch.nn as nn
from torchvision.models import resnet50, mobilenet_v3_large

from . import tinycifarnet, wideresnet
from . import ops_fused_sycl_train

def _build_resnet50_from_cfg(cfg: dict):
    num_classes = int(cfg["model"]["num_classes"])
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _build_mobilenet_v3_large_from_cfg(cfg: dict):
    num_classes = int(cfg["model"]["num_classes"])
    m = mobilenet_v3_large(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

_BUILDERS = {
    "tinycifarnet": tinycifarnet.build_model,       
    "tiny": tinycifarnet.build_model,               
    "wideresnet": wideresnet.build_model,
    "wrn": wideresnet.build_model,                  
    "resnet50": _build_resnet50_from_cfg,
    "mobilenet_v3_large": _build_mobilenet_v3_large_from_cfg,
    "mnetv3l": _build_mobilenet_v3_large_from_cfg,  
    "ops_tinycifarnet_sycl": ops_fused_sycl_train.build_model,
}

def build_model(name: str, cfg: dict):
    name = name.lower()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(_BUILDERS.keys())}")
    return _BUILDERS[name](cfg)

__all__ = [
    "build_model",
    "tinycifarnet",
    "wideresnet",
    "ops_fused_sycl_train",
]
