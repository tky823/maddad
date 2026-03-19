from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as aT

from ..functional.segment import segment


class BeatThisTransform(nn.Module):
    def __init__(self, chunk_size: int = 1500, pad: int = 6) -> None:
        super().__init__()

        self.spectrogram = BeatThisMelSpectrogram()

        self.chunk_size = chunk_size
        self.pad = pad

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, int]:
        x = self.spectrogram(waveform)
        output, last_offset = segment(x, chunk_size=self.chunk_size, pad=self.pad)

        return output, last_offset

    @property
    def sample_rate(self) -> int:
        return self.spectrogram.sample_rate

    @property
    def hop_length(self) -> int:
        return self.spectrogram.hop_length


class BeatThisMelSpectrogram(aT.MelSpectrogram):
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 441,
        f_min: float = 30,
        f_max: Optional[float] = 11000,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
        power: float = 1.0,
        normalized: bool = "frame_length",
        wkwargs: Optional[Dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "slaney",
        scale: float = 1000,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
            norm=norm,
            mel_scale=mel_scale,
        )

        self.scale = scale

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = super().forward(waveform)

        return torch.log1p(self.scale * spec)
