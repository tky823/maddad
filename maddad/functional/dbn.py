from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._C import decode_beat_peaks_by_viterbi as decode_beat_peaks_by_viterbi_cpp  # noqa: F401


@torch.no_grad()
def decode_beat_peaks_by_viterbi(
    logit: torch.Tensor,
    frame_rate: int,
    *,
    min_bpm: Optional[float] = 55.0,
    max_bpm: Optional[float] = 215.0,
    bpms: Optional[torch.Tensor] = None,
    weight: float = 100,
) -> None:
    """Search for best beat path through state space using a dynamic Bayesian network (DBN) approach.

    Args:
        observation (torch.Tensor): observation of shape (batch_size, num_frames).

    Returns:
        torch.Tensor: Section indices of shape (batch_size, num_peaks).

    """
    logit = logit.cpu()

    if bpms is None:
        if min_bpm is None:
            min_bpm = 55.0

        if max_bpm is None:
            max_bpm = 215.0

        min_fpb = round(60 * frame_rate / max_bpm)
        max_fpb = round(60 * frame_rate / min_bpm)
        fpbs = torch.arange(min_fpb, max_fpb + 1)
    else:
        assert min_bpm is None and max_bpm is None, (
            "If bpms is provided, min_bpm and max_bpm should be None."
        )

        fpbs = torch.round(60 * frame_rate / bpms)
        fpbs = fpbs.long()

    ratio = torch.abs(fpbs / fpbs.unsqueeze(dim=-1) - 1)
    log_transition_prob = F.log_softmax(-weight * ratio, dim=-1)

    peaks = _decode_beat_peaks_by_viterbi(
        logit=logit,
        fpbs=fpbs,
        log_transition_prob=log_transition_prob,
    )

    peak_indices = []

    for _peaks in peaks:
        _peak_indices = torch.nonzero(_peaks)
        _peak_indices = _peak_indices.squeeze(dim=-1)
        peak_indices.append(_peak_indices)

    peak_indices = nn.utils.rnn.pad_sequence(peak_indices, batch_first=True, padding_value=-1)

    return peak_indices


@torch.no_grad()
def _decode_beat_peaks_by_viterbi(
    logit: torch.Tensor,
    *,
    fpbs: Optional[torch.LongTensor] = None,
    log_transition_prob: Optional[float] = None,
) -> torch.Tensor:
    """Search for best beat path through the state space using a dynamic Bayesian network (DBN) approach.

    Args:
        observation (torch.Tensor): observation of shape (batch_size, num_frames).

    Returns:
        torch.Tensor: Section indices of shape (batch_size, num_frames).

    """
    if log_transition_prob is None:
        weight = 100
        ratio = torch.abs(fpbs / fpbs.unsqueeze(dim=-1) - 1)
        log_transition_prob = F.log_softmax(-weight * ratio, dim=-1)

    peaks = torch.ops.maddad.decode_beat_peaks_by_viterbi.default(logit, fpbs, log_transition_prob)

    return peaks
