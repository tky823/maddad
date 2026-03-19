from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.activation import NonDynamicallyQuantizableLinear
from torch.nn.modules.utils import _pair

from ..functional.activation import scaled_dot_product_attention
from .activation import RotaryPositionalMultiheadAttention as _RotaryPositionalMultiheadAttention
from .positional_encoding import RotaryPositionalEmbedding


class Frontend(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = (4, 3),
        stride: _size_2_t = (4, 1),
        bias: bool = False,
    ) -> None:
        super().__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = (kernel_size[0] - stride[0]) // 2, (kernel_size[1] - stride[1]) // 2

        self.norm1 = nn.BatchNorm1d(in_channels)
        self.conv2d = nn.Conv2d(
            1,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.norm1(input)
        x = x.unsqueeze(dim=-3)
        x = self.conv2d(x)
        x = self.norm2(x)
        output = self.activation(x)

        return output


class DualPathRoFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim_feedforward: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        kernel_size: _size_2_t = (2, 3),
        stride: _size_2_t = (2, 1),
        layer_norm_eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        nhead = 1

        layers = []

        for _ in range(num_layers):
            layers.append(
                DualPathRoFormerEncoderLayer(
                    in_channels,
                    2 * in_channels,
                    nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    stride=stride,
                    layer_norm_eps=layer_norm_eps,
                    rope_base=rope_base,
                    share_heads=share_heads,
                    batch_first=batch_first,
                    norm_first=norm_first,
                    bias=bias,
                    **factory_kwargs,
                )
            )

            in_channels *= 2
            nhead *= 2
            dim_feedforward *= 2

        self.layers = nn.ModuleList(layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        for layer in self.layers:
            x = layer(x)

        output = x

        return output


class DualPathRoFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        kernel_size: _size_2_t = (2, 3),
        stride: _size_2_t = (2, 1),
        layer_norm_eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        self.intra_roformer = RoFormerEncoderLayer(
            in_channels,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            rope_base=rope_base,
            share_heads=share_heads,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            **factory_kwargs,
        )
        self.inter_roformer = RoFormerEncoderLayer(
            in_channels,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            rope_base=rope_base,
            share_heads=share_heads,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            **factory_kwargs,
        )

        padding = (kernel_size[0] - stride[0]) // 2, (kernel_size[1] - stride[1]) // 2
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            **factory_kwargs,
        )

        self.norm = nn.BatchNorm2d(out_channels, eps=layer_norm_eps, **factory_kwargs)

        if isinstance(activation, str):
            activation = get_activation(activation)

        if activation is F.relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, num_bins, num_frames = input.size()
        x = input.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size * num_frames, num_bins, in_channels)
        x = self.intra_roformer(x)
        x = x.view(batch_size, num_frames, num_bins, in_channels)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * num_bins, num_frames, in_channels)
        x = self.inter_roformer(x)
        x = x.view(batch_size, num_bins, num_frames, in_channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.norm(x)
        output = self.activation(x)

        return output


class RoFormerEncoderLayer(nn.Module):
    """Encoder layer of RoFormer for BeatThis."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        self.self_attn = RotaryPositionalMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            base=rope_base,
            share_heads=share_heads,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=True, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=True, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = get_activation(activation)

        if activation is F.relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src (torch.Tensor): the sequence to the encoder layer.
            src_mask (torch.BoolTensor, optional): the mask for the src sequence.
            src_key_padding_mask (torch.BoolTensor, optional): the mask for the src keys
                per batch.
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise NotImplementedError("is_causal=True is not supported.")

        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout2(x)


class RotaryPositionalMultiheadAttention(_RotaryPositionalMultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: Optional[bool] = None,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        qdim: Optional[int] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        base: int = 10000,
        share_heads: bool = True,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super(nn.MultiheadAttention, self).__init__()

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )

        self.embed_dim = embed_dim
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.qdim), **factory_kwargs)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, self.qdim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias is None:
            bias = False

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

        self.gate = nn.Linear(
            embed_dim, num_heads, bias=True, **factory_kwargs
        )  # bias is always True
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")
        else:
            self.bias_k = self.bias_v = None

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        self._reset_parameters()

        self.rope = RotaryPositionalEmbedding(base=base, batch_first=batch_first)

        self.share_heads = share_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of RotaryPositionalMultiheadAttention.

        Args:
            query (torch.Tensor): Sequence of shape (batch_size, query_length, embed_dim)
                if ``batch_first=True``, otherwise (query_length, batch_size, embed_dim).
            key (torch.Tensor): Sequence of shape (batch_size, key_length, embed_dim)
                if ``batch_first=True``, otherwise (key_length, batch_size, embed_dim).
            key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (batch_size, key_length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (query_length, key_length) or
                (batch_size * num_heads, query_length, key_length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape
                    (batch_size, num_heads, query_length, key_length) if
                    ``average_attn_weights=True``, otherwise
                    (batch_size, query_length, key_length).

        """
        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias

        head_dim = embed_dim // num_heads

        if batch_first:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query = query.transpose(1, 0)
                    key = key.transpose(1, 0)
                    value = key
            else:
                query = query.transpose(1, 0)
                key = key.transpose(1, 0)
                value = value.transpose(1, 0)

        query_length, batch_size, _ = query.size()
        key_length, _, _ = key.size()

        if self._qkv_same_embed_dim:
            q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
                in_proj_weight, [embed_dim] * 3, dim=-2
            )
        else:
            q_proj_weight = self.q_proj_weight
            k_proj_weight = self.k_proj_weight
            v_proj_weight = self.v_proj_weight

        if self.in_proj_bias is None:
            q_proj_bias, k_proj_bias, v_proj_bias = None, None, None
        else:
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                in_proj_bias, [embed_dim] * 3, dim=0
            )

        q = F.linear(query, q_proj_weight, bias=q_proj_bias)
        k = F.linear(key, k_proj_weight, bias=k_proj_bias)
        v = F.linear(value, v_proj_weight, bias=v_proj_bias)

        q = self._apply_positional_embedding(q.contiguous())
        k = self._apply_positional_embedding(k.contiguous())

        q = q.view(query_length, batch_size, num_heads, head_dim)
        k = k.view(key_length, batch_size, num_heads, head_dim)
        v = v.view(key_length, batch_size, num_heads, head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        dropout_p = dropout if self.training else 0

        qkv, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            need_weights=need_weights,
        )

        x = self.gate(query)
        x = F.sigmoid(x.permute(1, 2, 0))
        qkv = qkv * x.unsqueeze(dim=-1)

        if batch_first:
            qkv = qkv.permute(0, 2, 1, 3).contiguous()
            qkv = qkv.view(batch_size, query_length, embed_dim)
        else:
            qkv = qkv.permute(2, 0, 1, 3).contiguous()
            qkv = qkv.view(query_length, batch_size, embed_dim)

        output = self.out_proj(qkv)

        if average_attn_weights and need_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not need_weights:
            attn_weights = None

        return output, attn_weights


def get_activation(activation: str) -> nn.Module:
    """Get activation module by str.

    Args:
        activation (str): Name of activation module.

    Returns:
        nn.Module: Activation module.

    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()

    raise RuntimeError(f"activation should be relu/gelu/elu, not {activation}")
