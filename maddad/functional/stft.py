from typing import Any

import torch
import torch.nn.functional as F


def stft(
    input: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 441,
    window: torch.Tensor = torch.hann_window,
    include_nyquist: bool = False,
) -> torch.Tensor:
    if window is None:
        _window = window
    elif isinstance(window, torch.Tensor):
        _window = window
    elif is_callable_torch_window(window):
        _window = window(n_fft, periodic=False)
    elif callable(window):
        _window = window(n_fft)
    else:
        raise ValueError(f"Unsupported window type: {type(window)}")

    num_samples = input.size(-1)
    num_frames = num_samples // hop_length
    left_padding = n_fft // 2
    right_padding = (num_frames - 1) * hop_length + n_fft - (num_samples + left_padding)
    x = F.pad(input, (left_padding, right_padding))

    output = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=_window,
        normalized=False,
        onesided=True,
        center=False,
        return_complex=True,
    )

    if not include_nyquist:
        output, _ = torch.split(output, [output.size(-2) - 1, 1], dim=-2)

    return output


def is_callable_torch_window(window: Any) -> bool:
    if (
        window is torch.hann_window
        or window is torch.hamming_window
        or window is torch.kaiser_window
        or window is torch.bartlett_window
        or window is torch.blackman_window
    ):
        return True
    else:
        return False
