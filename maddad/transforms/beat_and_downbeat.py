from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional.dbn import decode_beat_and_downbeat_peaks_by_viterbi


class DBNBeatAndDownbeatDecoder(nn.Module):
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

    def forward(
        self, beat_logit: torch.Tensor, downbeat_logit: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Decode logits into beat and downbeat indices.

        Args:
            beat_logit (torch.Tensor): Logit tensor for beats of shape (batch_size, num_frames).
            downbeat_logit (torch.Tensor): Logit tensor for downbeats of shape (batch_size, num_frames).

        Returns:
            torch.Tensor: Beat indices of shape (batch_size, num_beats).

        """
        beat_log_prob = F.logsigmoid(beat_logit)
        downbeat_log_prob = F.logsigmoid(downbeat_logit)
        nonbeat_log_prob = F.logsigmoid(-torch.max(beat_logit, downbeat_logit))

        return decode_beat_and_downbeat_peaks_by_viterbi(
            beat_log_prob=beat_log_prob,
            downbeat_log_prob=downbeat_log_prob,
            nonbeat_log_prob=nonbeat_log_prob,
            frame_rate=self.frame_rate,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            bpms=self.bpms,
            threshold=self.threshold,
        )
