import math
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
    threshold: Optional[float] = 0.05,
    weight: float = 100,
) -> torch.Tensor:
    """Search for best beat path through state space using a dynamic Bayesian network (DBN) approach.

    Args:
        logit (torch.Tensor): Logit of shape (batch_size, num_frames).
        frame_rate (int): Number of frames per second.
        min_bpm (float, optional): Minimum BPM. Defaults to ``55.0``.
        max_bpm (float, optional): Maximum BPM. Defaults to ``215.0``.
        bpms (torch.Tensor, optional): BPMs to consider. If provided, min_bpm and max_bpm \
            are ignored. Defaults to ``None``.
        threshold (float, optional): Threshold for beat activation. Defaults to ``0.05``.
        weight (float, optional): Weight for transition probability. Defaults to ``100``.

    Returns:
        torch.Tensor: Section indices of shape (batch_size, num_peaks). ``-1`` indicates padding.

    """
    device = logit.device
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

    batch_size, num_frames = logit.size()

    if threshold is None:
        offsets = torch.zeros(batch_size, dtype=torch.long)
        lengths = torch.full((batch_size,), num_frames, dtype=torch.long)
    else:
        assert 0 < threshold < 1, "Threshold should be between 0 and 1."

        padding_mask = logit < math.log(threshold / (1 - threshold))
        non_padding_mask = torch.logical_not(padding_mask)

        is_valid_sample = torch.any(non_padding_mask, dim=-1)
        is_invalid_sample = torch.logical_not(is_valid_sample)
        num_invalid_samples = torch.sum(is_invalid_sample).item()

        if num_invalid_samples > 0:
            raise ValueError(
                f"{num_invalid_samples} samples are invalid out of {batch_size} samples. Set larger threshold than ({threshold}) or None."
            )

        reversed_non_padding_mask = torch.flip(non_padding_mask, dims=(-1,))
        offsets = torch.argmax(non_padding_mask.long(), dim=-1)
        trimmings = torch.argmax(reversed_non_padding_mask.long(), dim=-1)

        trimmed_beat_log_prob = []
        trimmed_nonbeat_log_prob = []
        lengths = []

        for _logit, offset, trimming in zip(logit, offsets, trimmings):
            offset = offset.item()
            trimming = trimming.item()
            length = _logit.size(0) - offset - trimming
            _, _logit, _ = torch.split(
                _logit,
                [offset, num_frames - offset - trimming, trimming],
                dim=-1,
            )
            _beat_log_prob = F.logsigmoid(_logit)
            _nonbeat_log_prob = F.logsigmoid(-_logit)
            trimmed_beat_log_prob.append(_beat_log_prob)
            trimmed_nonbeat_log_prob.append(_nonbeat_log_prob)
            lengths.append(length)

        beat_log_prob = nn.utils.rnn.pad_sequence(
            trimmed_beat_log_prob, batch_first=True, padding_value=-float("inf")
        )
        nonbeat_log_prob = nn.utils.rnn.pad_sequence(
            trimmed_nonbeat_log_prob, batch_first=True, padding_value=-float("inf")
        )
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    peaks = _decode_beat_peaks_by_viterbi(
        beat_log_prob=beat_log_prob,
        nonbeat_log_prob=nonbeat_log_prob,
        lengths=lengths,
        fpbs=fpbs,
        log_transition_prob=log_transition_prob,
    )

    peak_indices = []

    for _peaks, offset in zip(peaks, offsets):
        _peak_indices = torch.nonzero(_peaks)
        _peak_indices = _peak_indices.squeeze(dim=-1)
        peak_indices.append(_peak_indices + offset)

    peak_indices = nn.utils.rnn.pad_sequence(peak_indices, batch_first=True, padding_value=-1)
    peak_indices = peak_indices.to(device)

    return peak_indices


@torch.no_grad()
def _decode_beat_peaks_by_viterbi(
    beat_log_prob: torch.Tensor,
    nonbeat_log_prob: torch.Tensor,
    *,
    lengths: Optional[torch.Tensor] = None,
    fpbs: Optional[torch.LongTensor] = None,
    log_transition_prob: Optional[float] = None,
) -> torch.Tensor:
    """Search for best beat path through the state space using a dynamic Bayesian network (DBN) approach.

    Args:
        beat_log_prob (torch.Tensor): Log probability for beat of shape (batch_size, num_frames).
        nonbeat_log_prob (torch.Tensor): Log probability for non-beat of shape (batch_size, num_frames).
        lengths (torch.Tensor, optional): Lengths of each sample in the batch. If not provided, \
            it is assumed that all samples have the same length. Defaults to ``None``.
        fpbs (torch.Tensor): Frames per beat to assume.
        log_transition_prob (torch.Tensor, optional): Log transition probability of shape (num_fpbs, num_fpbs). \
            If not provided, it will be computed from fpbs based on official implementation.

    Returns:
        torch.Tensor: Section indices of shape (batch_size, num_frames).

    """
    if lengths is None:
        batch_size, num_frames = beat_log_prob.size()
        lengths = torch.full((batch_size,), num_frames, dtype=torch.long)

    if fpbs is None:
        raise ValueError("fpbs must be provided.")

    if log_transition_prob is None:
        weight = 100
        ratio = torch.abs(fpbs / fpbs.unsqueeze(dim=-1) - 1)
        log_transition_prob = F.log_softmax(-weight * ratio, dim=-1)

    peaks = torch.ops.maddad.decode_beat_peaks_by_viterbi.default(
        beat_log_prob, nonbeat_log_prob, lengths, fpbs, log_transition_prob
    )

    return peaks
