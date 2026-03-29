from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional.dbn import decode_beat_peaks_by_viterbi


class DBNBeatDecoder(nn.Module):
    def __init__(
        self,
        frame_rate: int,
        *,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        bpms: List[float] = None,
        threshold: Optional[float] = 0.05,
    ) -> None:
        super().__init__()

        self.frame_rate = frame_rate
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.bpms = bpms
        self.threshold = threshold

    def forward(self, logit: torch.Tensor) -> torch.Tensor:
        """Decode logit into beat indices.

        Args:
            logit (torch.Tensor): Logit tensor of shape (batch_size, num_frames).

        Returns:
            torch.Tensor: Beat indices of shape (batch_size, num_beats).

        """
        beat_log_prob = F.logsigmoid(logit)
        nonbeat_log_prob = F.logsigmoid(-logit)

        return decode_beat_peaks_by_viterbi(
            beat_log_prob=beat_log_prob,
            nonbeat_log_prob=nonbeat_log_prob,
            frame_rate=self.frame_rate,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            bpms=self.bpms,
            threshold=self.threshold,
        )
