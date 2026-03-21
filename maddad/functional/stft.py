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
    """Wrapper function of torch.stft compatible with madmom.

    Args:
        input (torch.Tensor): Waveform of shape (*, num_samples).
        n_fft (int): FFT window size.
        hop_length (int): Hop length.
        window (torch.Tensor or callable, optional): torch window function (e.g., ``torch.hann_window``), \
            or callable that takes n_fft as argument and returns window tensor.
        include_nyquist: Whether to include Nyquist frequency bin in output.

    Returns:
        torch.Tensor: STFT of shape (*, num_bins, num_frames), where num_bins is n_fft // 2 + 1 \
            if ``include_nyquist`` is ``True``, else n_fft // 2.

    """
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
    num_frames = (num_samples - 1) // hop_length + 1
    padding = (num_frames - 1) * hop_length + n_fft - num_samples
    left_padding = n_fft // 2
    right_padding = padding - left_padding
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
