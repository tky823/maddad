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
        """STFT class compatible with madmom.

        Args:
            n_fft (int): FFT window size.
            hop_length (int): Hop length.
            window (torch.Tensor or callable, optional): torch window function (e.g., ``torch.hann_window``), \
                or callable that takes n_fft as argument and returns window tensor.
            include_nyquist: Whether to include Nyquist frequency bin in output.

        """
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
        """Forward pass of STFT.

        Args:
            input (torch.Tensor): Waveform of shape (*, num_samples).

        Returns:
            torch.Tensor: STFT of shape (*, num_bins, num_frames), where num_bins is n_fft // 2 + 1 \
                if ``include_nyquist`` is ``True``, else n_fft // 2.

        """
        return stft(input, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)


class STFT(ShortTimeFourierTransform):
    """Alias for ShortTimeFourierTransform."""

    pass
