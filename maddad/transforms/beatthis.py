from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MinimalBeatDecoder(nn.Module):
    def __init__(self, pool_size: int = 7, merge_interval: int = 1) -> None:
        super().__init__()

        assert pool_size % 2 == 1, "pool_size must be odd"

        self.pool_size = pool_size
        self.merge_interval = merge_interval

    def forward(self, logit: torch.Tensor) -> torch.Tensor:
        """Decode logit into beat indices.

        Args:
            logit (torch.Tensor): Logit tensor of shape (1, num_frames).

        Returns:
            torch.Tensor: Beat indices of shape (1, num_beats).

        .. note::

            Only batch_size=1 is supported for now.

        .. note::

            Due to averaging, output beat indices may not be integers.

        """
        pool_size = self.pool_size
        stride = 1
        padding = pool_size // 2

        assert logit.dim() == 2 and logit.size(0) == 1, "Only batch_size=1 is supported for now."

        pooled_logit = F.max_pool1d(
            logit,
            kernel_size=pool_size,
            stride=stride,
            padding=padding,
        )
        logit = logit.squeeze(dim=0)
        pooled_logit = pooled_logit.squeeze(dim=0)
        pooled_logit = pooled_logit.masked_fill(logit != pooled_logit, -float("inf"))
        peaks = torch.nonzero(pooled_logit > 0, as_tuple=False)
        peaks = peaks.squeeze(dim=-1)
        peak_intervals = torch.diff(peaks, dim=-1)
        is_new_section = F.pad(peak_intervals > self.merge_interval, (1, 0), value=True)
        sections = torch.cumsum(is_new_section.long(), dim=-1) - 1
        num_sections = sections[-1]
        num_sections = num_sections.item() + 1
        sum_peaks = torch.zeros((num_sections,), dtype=torch.long, device=peaks.device)
        count_peaks = torch.zeros((num_sections,), dtype=torch.long, device=peaks.device)
        sum_peaks.scatter_add_(0, sections, peaks)
        count_peaks.scatter_add_(0, sections, torch.ones_like(peaks, dtype=torch.long))
        output = sum_peaks / count_peaks
        output = output.unsqueeze(dim=0)

        return output


class MinimalBeatAndDownbeatDecoder(nn.Module):
    def __init__(self, pool_size: int = 7, merge_interval: int = 1) -> None:
        super().__init__()

        assert pool_size % 2 == 1, "pool_size must be odd"

        self.pool_size = pool_size
        self.merge_interval = merge_interval

    def forward(
        self, beat_logit: torch.Tensor, downbeat_logit: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Decode logit into beat indices.

        Args:
            beat_logit (torch.Tensor): Logit of beats of shape (1, num_frames).
            downbeat_logit (torch.Tensor): Logit of downbeats of shape (1, num_frames).

        Returns:
            tuple: Tuple of tensors containing:
                - torch.Tensor: Beat indices of shape (1, num_beats).
                - torch.LongTensor: Beat indices of shape (1, num_beats).

        .. note::

            Only batch_size=1 is supported for now.

        .. note::

            Due to averaging, output beat indices may not be integers.

        """
        pool_size = self.pool_size
        stride = 1
        padding = pool_size // 2

        assert beat_logit.dim() == 2 and beat_logit.size(0) == 1, (
            "Only batch_size=1 is supported for now."
        )
        assert downbeat_logit.dim() == 2 and downbeat_logit.size(0) == 1, (
            "Only batch_size=1 is supported for now."
        )

        device = beat_logit.device

        logit = torch.stack([beat_logit, downbeat_logit], dim=-2)
        pooled_logit = F.max_pool1d(
            logit,
            kernel_size=pool_size,
            stride=stride,
            padding=padding,
        )
        logit = logit.squeeze(dim=0)
        pooled_logit = pooled_logit.squeeze(dim=0)
        pooled_logit = pooled_logit.masked_fill(logit != pooled_logit, -float("inf"))
        pooled_beat_logit, pooled_downbeat_logit = pooled_logit.unbind(dim=-2)
        beat_peaks = torch.nonzero(pooled_beat_logit > 0, as_tuple=False)
        downbeat_peaks = torch.nonzero(pooled_downbeat_logit > 0, as_tuple=False)
        beat_peaks = beat_peaks.squeeze(dim=-1)
        downbeat_peaks = downbeat_peaks.squeeze(dim=-1)
        beat_peak_intervals = torch.diff(beat_peaks, dim=-1)
        downbeat_peak_intervals = torch.diff(downbeat_peaks, dim=-1)

        # beat
        is_new_beat_section = F.pad(beat_peak_intervals > self.merge_interval, (1, 0), value=True)
        beat_sections = torch.cumsum(is_new_beat_section.long(), dim=-1) - 1
        num_beat_sections = beat_sections[-1]
        num_beat_sections = num_beat_sections.item() + 1
        sum_beat_peaks = torch.zeros((num_beat_sections,), dtype=torch.long, device=device)
        count_beat_peaks = torch.zeros((num_beat_sections,), dtype=torch.long, device=device)
        sum_beat_peaks.scatter_add_(0, beat_sections, beat_peaks)
        count_beat_peaks.scatter_add_(
            0, beat_sections, torch.ones_like(beat_peaks, dtype=torch.long)
        )
        output = sum_beat_peaks / count_beat_peaks

        # downbeat
        is_new_downbeat_section = F.pad(
            downbeat_peak_intervals > self.merge_interval, (1, 0), value=True
        )
        downbeat_sections = torch.cumsum(is_new_downbeat_section.long(), dim=-1) - 1
        num_downbeat_sections = downbeat_sections[-1]
        num_downbeat_sections = num_downbeat_sections.item() + 1
        sum_downbeat_peaks = torch.zeros((num_downbeat_sections,), dtype=torch.long, device=device)
        count_downbeat_peaks = torch.zeros(
            (num_downbeat_sections,), dtype=torch.long, device=device
        )
        sum_downbeat_peaks.scatter_add_(0, downbeat_sections, downbeat_peaks)
        count_downbeat_peaks.scatter_add_(
            0, downbeat_sections, torch.ones_like(downbeat_peaks, dtype=torch.long)
        )
        downbeat_output = sum_downbeat_peaks / count_downbeat_peaks

        # align downbeat with beat
        distance = torch.abs(downbeat_output.unsqueeze(dim=-1) - output)
        nearest_indices = torch.argmin(distance, dim=-1)
        nearest_indices = torch.unique(nearest_indices)
        first_downbeat = nearest_indices[0]
        last_downbeat = nearest_indices[-1]
        beats = torch.diff(nearest_indices, dim=-1)

        # leading & trailing beats
        first_meter = torch.max(beats[0], first_downbeat)
        first_meter = torch.clip(first_meter, min=2)
        leading_beats = first_meter - torch.arange(first_downbeat, device=device)
        last_meter = output.size(-1) - last_downbeat
        trailing_beats = torch.arange(last_meter, device=device) + 1

        # beats
        bar_offset = torch.cumsum(beats, dim=-1)
        bar_offset = F.pad(bar_offset, (1, -1))
        offset = torch.repeat_interleave(bar_offset, beats)
        num_bars = beats.sum(dim=-1)
        indices = torch.arange(num_bars, device=device) - offset + 1
        indices = torch.cat([leading_beats, indices, trailing_beats], dim=-1)

        output = output.unsqueeze(dim=0)
        indices = indices.unsqueeze(dim=0)

        return output, indices
