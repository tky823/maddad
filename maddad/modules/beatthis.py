from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import NonDynamicallyQuantizableLinear

from ..functional.activation import scaled_dot_product_attention
from .activation import RotaryPositionalMultiheadAttention as _RotaryPositionalMultiheadAttention
from .positional_encoding import RotaryPositionalEmbedding


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

        self.gate = nn.Linear(qdim, num_heads, bias=True, **factory_kwargs)  # bias is always True
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, qdim, bias=bias, **factory_kwargs
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
