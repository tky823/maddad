import torch
import torch.nn as nn

from ..functional.positional_encoding import rotary_positional_embedding

__all__ = [
    "RotaryPositionalEmbedding",
    "RoPE",
]


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: Rotary positional embedding proposed in [#su2021roformer]_.

    .. [#su2021roformer]
        J. Su et al., "RoFormer: Enhanced transformer with rotary position embedding,"
        *Neurocomputing*, vol. 568, 2024.

    """

    def __init__(
        self,
        base: int = 10000,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.base = base
        self.batch_first = batch_first

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of RoPE.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).

        Returns:
            torch.Tensor: Sequence of same shape as input.

        """
        output = rotary_positional_embedding(input, base=self.base, batch_first=self.batch_first)

        return output


class RoPE(RotaryPositionalEmbedding):
    """Alias of RotaryPositionalEmbedding."""
