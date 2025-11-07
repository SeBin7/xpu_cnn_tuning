import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class WideBasic(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_rate=0.0):
        super().__init__()
        self.drop_rate = drop_rate

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = out + self.shortcut(x)
        out = self.relu2(out)
        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        depth=28,
        widen_factor=10,
        dropout=0.0,
        num_classes=10,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth should be 6n+4 (e.g., 16, 28, 40).")
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = conv3x3(3, widths[0])
        self.block1 = self._make_layer(widths[0], widths[1], n, stride=1, drop_rate=dropout)
        self.block2 = self._make_layer(widths[1], widths[2], n, stride=2, drop_rate=dropout)
        self.block3 = self._make_layer(widths[2], widths[3], n, stride=2, drop_rate=dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride, drop_rate):
        layers = []
        layers.append(WideBasic(in_channels, out_channels, stride=stride, drop_rate=drop_rate))
        for _ in range(1, blocks):
            layers.append(WideBasic(out_channels, out_channels, stride=1, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def build_model(cfg):
    model_cfg = cfg["model"]
    depth = model_cfg.get("wrn_depth", 28)
    widen = model_cfg.get("wrn_width", 10)
    dropout = model_cfg.get("wrn_dropout", cfg["train"].get("dropout", 0.0))
    num_classes = model_cfg["num_classes"]
    return WideResNet(depth=depth, widen_factor=widen, dropout=dropout, num_classes=num_classes)
