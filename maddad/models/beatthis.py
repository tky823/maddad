import os
from typing import Dict, OrderedDict, Tuple, Union

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.modules.utils import _pair

from ..modules.beatthis import (
    BeatDownbeatHead,
    DualPathRoFormerEncoder,
    Encoder,
    Frontend,
    Projector,
    RoFormerEncoderLayer,
)
from ..utils._github import download_file_from_github_release


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
        if hasattr(nn, "RMSNorm"):
            rms_norm_cls = nn.RMSNorm
        else:
            from ..modules.normalization import RMSNorm

            rms_norm_cls = RMSNorm

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
        backbone_norm = rms_norm_cls(backbone_d_model, eps=layer_norm_eps)
        backbone = nn.TransformerEncoder(layer, num_layers=num_backbone_layers, norm=backbone_norm)
        head = BeatDownbeatHead(backbone_d_model)

        return cls(encoder, backbone, head)

    @classmethod
    def build_from_pretrained(
        cls, pretrained_model_name_or_path: str = "official_beatthis"
    ) -> "BeatThis":

        pretrained_model_configs = _create_pretrained_beatthis_configs()

        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            model_state_dict: OrderedDict = state_dict["model"]
            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)
            pretrained_model_config = resolved_config.model
            model: BeatThis = hydra.utils.instantiate(pretrained_model_config)
            model.load_state_dict(model_state_dict)

            return model
        elif pretrained_model_name_or_path in pretrained_model_configs:
            config = pretrained_model_configs[pretrained_model_name_or_path]
            url = config["url"]
            path = config["path"]
            download_file_from_github_release(url, path=path)
            model = cls.build_from_pretrained(path)

            return model
        else:
            raise FileNotFoundError(f"{pretrained_model_name_or_path} does not exist.")


def _create_pretrained_beatthis_configs() -> Dict[str, Dict[str, str]]:
    """Create pretrained_model_configs without circular import error."""

    from ..utils import model_cache_dir

    pretrained_model_configs = {
        "official_beatthis": {
            "url": "https://github.com/tky823/maddad/releases/download/v0.0.0/official_beatthis.pth",  # noqa: E501
            "path": os.path.join(
                model_cache_dir,
                "BeatThis",
                "5756c850",
                "official_beatthis.pth",
            ),
            "sha256": "5756c850d9e4fc7aa8c81eb8e18ca3f341c369fcbaa255dadf3a623d8546683b",
        },
    }

    return pretrained_model_configs
