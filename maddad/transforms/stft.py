from typing import Optional

import torch
import torch.nn as nn

from ..functional.stft import is_callable_torch_window, stft


class ShortTimeFourierTransform(nn.Module):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        window: Optional[torch.Tensor] = torch.hann_window,
    ) -> None:
        super().__init__()

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

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer("window", _window)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return stft(input, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)


class STFT(ShortTimeFourierTransform):
    """Alias for ShortTimeFourierTransform."""

    pass
