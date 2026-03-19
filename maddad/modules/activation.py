from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from ..functional.activation import scaled_dot_product_attention
from .positional_encoding import RotaryPositionalEmbedding

IS_TORCH_LT_2_0 = version.parse(torch.__version__) < version.parse("2.0")


class RotaryPositionalMultiheadAttention(nn.MultiheadAttention):
    """Multihead attention using rotary positional representation.

    The implementation is ported from
    https://github.com/tky823/Audyn/blob/c1aed30b3ce09d94ea76029416fe392efe9cf209/audyn/modules/activation.py#L1039-L1236
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        base: int = 10000,
        share_heads: bool = True,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            **factory_kwargs,
        )

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

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

    def _apply_positional_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input.

        Args:
            input (torch.Tensor): Sequence of shape (length, batch_size, num_heads, head_dim).

        Returns:
            torch.Tensor: Output sequence same shape as input.

        """
        share_heads = self.share_heads
        num_heads = self.num_heads
        input_length, batch_size, embed_dim = input.size()
        head_dim = embed_dim // num_heads

        if share_heads:
            x = input.view(input_length, batch_size * num_heads, head_dim)
        else:
            x = input.view(input_length, batch_size, num_heads * head_dim)

        if self.batch_first:
            x = x.transpose(1, 0)

        x = self.rope(x)

        if self.batch_first:
            x = x.transpose(1, 0).contiguous()

        output = x.view(input_length, batch_size, embed_dim)

        return output

    def validate_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate keyword arguments for backward compatibility."""
        valid_keys = set()

        if not IS_TORCH_LT_2_0:
            valid_keys.add("is_causal")

        invalid_keys = set(kwargs.keys()) - valid_keys

        assert invalid_keys == set(), f"Invalid keys {invalid_keys} are given."
