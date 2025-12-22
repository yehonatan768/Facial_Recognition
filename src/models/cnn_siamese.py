from __future__ import annotations

from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn


class SiameseCNN(nn.Module):
    """
    The paper's siamese convolutional tower (one twin).
    This module is used twice (shared weights) to produce h1, h2.

    Figure 4 (best conv architecture) is:
      Input: 1 x 105 x 105
      Conv(64, 10x10) + ReLU, valid, stride=1
      MaxPool(2x2, stride=2)
      Conv(128, 7x7) + ReLU
      MaxPool(2x2, stride=2)
      Conv(128, 4x4) + ReLU
      MaxPool(2x2, stride=2)
      Conv(256, 4x4) + ReLU
      Flatten
      FC(4096) + Sigmoid   (paper states last layers are sigmoidal)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()

        mcfg = cfg["model"]
        input_channels = int(mcfg["input_channels"])
        input_size = int(mcfg["input_size"])

        conv_spec: List[Tuple[int, int, bool]] = [
            (int(x["out_channels"]), int(x["kernel"]), bool(x["maxpool"]))
            for x in mcfg["cnn"]["layers"]
        ]
        fc_units = int(mcfg["cnn"]["fc_units"])

        layers: List[nn.Module] = []
        c_in = input_channels
        for out_ch, k, do_pool in conv_spec:
            layers.append(nn.Conv2d(c_in, out_ch, kernel_size=k, stride=1, padding=0))  # valid conv
            layers.append(nn.ReLU(inplace=True))
            if do_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            c_in = out_ch

        self.conv = nn.Sequential(*layers)

        # infer flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            y = self.conv(dummy)
            flat_dim = int(y.numel())

        self.fc = nn.Linear(flat_dim, fc_units)
        self.fc_act = nn.Sigmoid()  # paper: "sigmoidal units in the remaining layers"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.fc_act(x)
        return x


def init_weights_like_paper(model: nn.Module, cfg: Dict[str, Any]) -> None:
    """
    Paper initialization:
      - Conv weights ~ N(0, 1e-2); conv biases ~ N(0.5, 1e-2)
      - FC weights ~ N(0, 2e-1);  FC biases  ~ N(0.5, 1e-2)
    """
    init_cfg = cfg["train"]["init"]

    conv_w_std = float(init_cfg["conv_weight_std"])
    conv_b_mean = float(init_cfg["bias_mean"])
    conv_b_std = float(init_cfg["bias_std"])
    fc_w_std = float(init_cfg["fc_weight_std"])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=conv_w_std)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=conv_b_mean, std=conv_b_std)

        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=fc_w_std)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=conv_b_mean, std=conv_b_std)
