from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ..modules.beatthis import (
    BeatDownbeatHead,
    DualPathRoFormerEncoder,
    Encoder,
    Frontend,
    Projector,
    RoFormerEncoderLayer,
)


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

    @classmethod
    def build_from_config(cls, version: str = "default") -> "BeatThis":
        if version == "default":
            num_bins = 128

            layer_norm_eps = 1e-5
            share_heads = True
            batch_first = True
            norm_first = True
            bias = False

            # encoder
            dim_encoder_feedforward = 128
            num_encoder_features = 32
            num_encoder_layers = 3
            encoder_dropout = 0.1

            # backbone
            backbone_d_model = 512
            backbone_nhead = 16
            backbone_dim_feedforward = 2048
            backbone_dropout = 0.2
            num_backbone_layers = 6
        else:
            raise ValueError(f"Unsupported version: {version}")

        encoder_frontend = Frontend(num_bins, num_encoder_features)
        encoder_backbone = DualPathRoFormerEncoder(
            num_encoder_features,
            dim_encoder_feedforward,
            num_layers=num_encoder_layers,
            activation=F.gelu,
            layer_norm_eps=layer_norm_eps,
            dropout=encoder_dropout,
            share_heads=share_heads,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
        )
        _stride, _ = _pair(encoder_frontend.conv2d.stride)
        encoder_proj = Projector(num_bins // _stride * num_encoder_features, backbone_d_model)
        encoder = Encoder(encoder_frontend, encoder_backbone, encoder_proj)
        layer = RoFormerEncoderLayer(
            backbone_d_model,
            backbone_nhead,
            dim_feedforward=backbone_dim_feedforward,
            dropout=backbone_dropout,
            activation=F.gelu,
            layer_norm_eps=layer_norm_eps,
            share_heads=share_heads,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
        )
        backbone_norm = nn.RMSNorm(backbone_d_model, eps=layer_norm_eps)
        backbone = nn.TransformerEncoder(layer, num_layers=num_backbone_layers, norm=backbone_norm)
        head = BeatDownbeatHead(backbone_d_model)

        return cls(encoder, backbone, head)
