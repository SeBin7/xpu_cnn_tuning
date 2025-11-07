from torch import nn
from .ops_fused_train import FusedConv3x3BNFoldReLUTrain

class TinyCIFARNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem  = FusedConv3x3BNFoldReLUTrain(3, 32)
        self.block1= FusedConv3x3BNFoldReLUTrain(32, 64)
        self.block2= FusedConv3x3BNFoldReLUTrain(64, 128)
        self.head  = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)

import torch.nn as nn
from .ops_fused_train import FusedConv3x3BNFoldReLUTrain

class TinyCIFARNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem   = FusedConv3x3BNFoldReLUTrain(3, 32)
        self.block1 = FusedConv3x3BNFoldReLUTrain(32, 64)
        self.block2 = FusedConv3x3BNFoldReLUTrain(64, 128)
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.stem(x); x = self.block1(x); x = self.block2(x)
        return self.head(x)

def build_model(cfg: dict) -> nn.Module:
    # Optional: warm-load the fused .so once (no-op if already loaded)
    try:
        from fused_ops import load_fused  # because we run as "python src/train_xpu.py"
        load_fused()
    except Exception:
        pass
    num_classes = int(cfg["model"]["num_classes"])
    return TinyCIFARNet(num_classes=num_classes)