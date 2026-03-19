from typing import Tuple, Union

import torch
import torch.nn as nn


class BeatThis(nn.Module):
    def __init__(self, encoder: nn.Module, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.backbone = backbone
        self.head = head

    def forward(
        self, input: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.encoder(input)
        x = self.backbone(x)
        output = self.head(x)

        return output
