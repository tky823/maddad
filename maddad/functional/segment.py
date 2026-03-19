import torch
import torch.nn.functional as F


def segment(input: torch.Tensor, chunk_size: int, pad: int = 6) -> torch.Tensor:
    """Segment input into overlapping chunks.

    Args:
        input (torch.Tensor): Tensor of shape (*, length)
        chunk_size (int): Size of each chunk.
        pad (int): Amount of padding on each side of sequence. Default: ``6``.

    Returns:
        torch.Tensor: Tensor of shape (num_chunks, *, chunk_size).

    """

    *batch_shape, length = input.size()
    hop_size = chunk_size - 2 * pad

    x = input.view(-1, 1, 1, length)
    x = F.pad(x, (pad, 0))
    x = F.unfold(x, kernel_size=(1, chunk_size), stride=(1, hop_size))
    x = x.permute(2, 0, 1).contiguous()
    output = x.view(-1, *batch_shape, chunk_size)

    if (length + pad - chunk_size) % hop_size > 0:
        # TODO: corner case
        _, _x = torch.split(input, [length + pad - chunk_size, chunk_size - pad], dim=-1)
        _x = F.pad(_x, (0, pad))
        _x = _x.unsqueeze(dim=0)
        output = torch.cat([output, _x], dim=0)

    return output
