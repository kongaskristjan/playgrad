"""Small CIFAR-style residual network.

The architecture follows the original CIFAR ResNet from He et al. (2015):
an initial 3x3 conv followed by three stages of basic residual blocks
with channel widths (16, 32, 64) and spatial downsampling between stages.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BasicBlock(nn.Module):
    """Two 3x3 convs with a residual skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNetCIFAR(nn.Module):
    """Small CIFAR ResNet with `blocks_per_stage` BasicBlocks at each of 3 stages.

    Total depth = 6 * blocks_per_stage + 2 (e.g. ResNet-20 -> blocks_per_stage=3).
    """

    def __init__(self, num_classes: int = 10, blocks_per_stage: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(16, 16, blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(16, 32, blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(32, 64, blocks_per_stage, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(num_classes=num_classes, blocks_per_stage=3)
