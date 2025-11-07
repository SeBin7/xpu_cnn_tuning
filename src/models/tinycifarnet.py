import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops_fused_sycl_train import FusedConv3x3BNReLUTrainS

def conv3x3(in_c, out_c, s=1):
    return nn.Conv2d(in_c, out_c, 3, stride=s, padding=1, bias=False)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, down=False, use_fused=True):
        super().__init__()
        s = 2 if down else 1
        if use_fused:
            self.net = nn.Sequential(
                FusedConv3x3BNReLUTrainS(in_c,  out_c, stride=s, padding=1, bias=True),
                FusedConv3x3BNReLUTrainS(out_c, out_c, stride=1, padding=1, bias=True),
            )
        else:
            self.net = nn.Sequential(
                conv3x3(in_c, out_c, s), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                conv3x3(out_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.net(x)

class ResidualFFN(nn.Module):
    def __init__(self, dim, hidden, p=0.1, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, dim), nn.Dropout(p),
        )
    def forward(self, x):
        if self.pre_norm:
            return x + self.ff(self.norm(x))
        y = x + self.ff(x)
        return self.norm(y)

class TinyCIFARNet(nn.Module):
    def __init__(self, num_classes=10, widths=(64,128,256,256), ffn_expand=4, dropout=0.1, use_fused=True):
        super().__init__()
        c1, c2, c3, c4 = widths
        if use_fused:
            self.stem = FusedConv3x3BNReLUTrainS(3, c1, stride=1, padding=1, bias=True)
        else:
            self.stem = nn.Sequential(conv3x3(3, c1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.b1 = ConvBlock(c1, c1, down=False, use_fused=use_fused)  # 출력 해상도 32x32
        self.b2 = ConvBlock(c1, c2, down=True,  use_fused=use_fused)  # 출력 해상도 16x16
        self.b3 = ConvBlock(c2, c3, down=True,  use_fused=use_fused)  # 출력 해상도 8x8
        self.b4 = ConvBlock(c3, c4, down=True,  use_fused=use_fused)  # 출력 해상도 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = c4
        self.ffn = ResidualFFN(feat_dim, ffn_expand * feat_dim, p=dropout, pre_norm=True)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = self.pool(x).flatten(1)
        x = self.ffn(x)
        logits = self.classifier(x)
        return logits

def build_model(cfg):
    use_fused = cfg["model"].get("use_fused", True)
    return TinyCIFARNet(
        num_classes=cfg["model"]["num_classes"],
        widths=tuple(cfg["model"]["widths"]),
        ffn_expand=cfg["model"]["ffn_expand"],
        dropout=cfg["train"]["dropout"],
        use_fused=use_fused,
    )
