import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._C import (
    decode_beat_and_downbeat_peaks_by_viterbi as decode_beat_and_downbeat_peaks_by_viterbi_cpp,  # noqa: F401
)
from .._C import decode_beat_peaks_by_viterbi as decode_beat_peaks_by_viterbi_cpp  # noqa: F401


@torch.no_grad()
def decode_beat_peaks_by_viterbi(
    beat_log_prob: torch.Tensor,
    nonbeat_log_prob: torch.Tensor,
    frame_rate: int,
    *,
    min_bpm: Optional[float] = 55.0,
    max_bpm: Optional[float] = 215.0,
    bpms: Optional[torch.Tensor] = None,
    beat_region: float = 0.0625,
    threshold: Optional[float] = 0.05,
    weight: float = 100,
) -> torch.LongTensor:
    """Search for best beat path through state space using a dynamic Bayesian network (DBN) approach.

    Args:
        beat_log_prob (torch.Tensor): Log-probability of beat states of shape (batch_size, num_frames).
        nonbeat_log_prob (torch.Tensor): Log-probability of non-beat states of shape (batch_size, num_frames).
        frame_rate (int): Number of frames per second.
        min_bpm (float, optional): Minimum BPM. Defaults to ``55.0``.
        max_bpm (float, optional): Maximum BPM. Defaults to ``215.0``.
        bpms (torch.Tensor, optional): BPMs to consider. If provided, min_bpm and max_bpm \
            are ignored. Defaults to ``None``.
        threshold (float, optional): Threshold for beat activation. Defaults to ``0.05``.
        weight (float, optional): Weight for transition probability. Defaults to ``100``.

    Returns:
        torch.LongTensor: Section indices of shape (batch_size, num_peaks). ``-1`` indicates padding.

    """
    device = beat_log_prob.device
    beat_log_prob = beat_log_prob.cpu()
    nonbeat_log_prob = nonbeat_log_prob.cpu()

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

    batch_size, num_frames = beat_log_prob.size()

    if threshold is None:
        offsets = torch.zeros(batch_size, dtype=torch.long)
        lengths = torch.full((batch_size,), num_frames, dtype=torch.long)
    else:
        assert 0 < threshold < 1, "Threshold should be between 0 and 1."

        padding_mask = beat_log_prob < math.log(threshold)
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

        for _beat_log_prob, _nonbeat_log_prob, offset, trimming in zip(
            beat_log_prob, nonbeat_log_prob, offsets, trimmings
        ):
            offset = offset.item()
            trimming = trimming.item()
            length = _beat_log_prob.size(0) - offset - trimming
            sections = [offset, num_frames - offset - trimming, trimming]
            _, _beat_log_prob, _ = torch.split(_beat_log_prob, sections, dim=-1)
            _, _nonbeat_log_prob, _ = torch.split(_nonbeat_log_prob, sections, dim=-1)
            trimmed_beat_log_prob.append(_beat_log_prob)
            trimmed_nonbeat_log_prob.append(_nonbeat_log_prob)
            lengths.append(length)

        beat_log_prob = nn.utils.rnn.pad_sequence(
            trimmed_beat_log_prob, batch_first=True, padding_value=-float("inf")
        )
        nonbeat_log_prob = nn.utils.rnn.pad_sequence(
            trimmed_nonbeat_log_prob, batch_first=True, padding_value=-float("inf")
        )
        lengths = torch.tensor(lengths, dtype=torch.long)

    peaks = _decode_beat_peaks_by_viterbi(
        beat_log_prob=beat_log_prob,
        nonbeat_log_prob=nonbeat_log_prob,
        lengths=lengths,
        fpbs=fpbs,
        beat_region=beat_region,
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
def decode_beat_and_downbeat_peaks_by_viterbi(
    beat_log_prob: torch.Tensor,
    downbeat_log_prob: torch.Tensor,
    nonbeat_log_prob: torch.Tensor,
    frame_rate: int,
    *,
    min_bpm: Optional[float] = 55.0,
    max_bpm: Optional[float] = 215.0,
    bpms: Optional[torch.Tensor] = None,
    meters: List[int] = [3, 4],
    beat_region: float = 0.0625,
    threshold: Optional[float] = 0.05,
    weight: float = 100,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Search for best beat path through state space using a dynamic Bayesian network (DBN) approach.

    Args:
        beat_log_prob (torch.Tensor): Log-probability of beat states of shape (batch_size, num_frames).
        downbeat_log_prob (torch.Tensor): Log-probability of downbeat states of shape (batch_size, num_frames).
        nonbeat_log_prob (torch.Tensor): Log-probability of non-beat states of shape (batch_size, num_frames).
        frame_rate (int): Number of frames per second.
        min_bpm (float, optional): Minimum BPM. Defaults to ``55.0``.
        max_bpm (float, optional): Maximum BPM. Defaults to ``215.0``.
        bpms (torch.Tensor, optional): BPMs to consider. If provided, min_bpm and max_bpm \
            are ignored. Defaults to ``None``.
        meters (list): Meters to consider. Defaults to ``[3, 4]``.
        threshold (float, optional): Threshold for beat activation. Defaults to ``0.05``.
        weight (float, optional): Weight for transition probability. Defaults to ``100``.

    Returns:
        tuple: Tuple of tensors containing:
            - torch.LongTensor: Section indices of shape (batch_size, num_peaks). ``-1`` indicates padding.
            - torch.LongTensor: Beat indices of shape (batch_size, num_peaks). ``1`` corresponds to downbeat. ``-1`` indicates padding.

    """
    device = beat_log_prob.device
    beat_log_prob = beat_log_prob.cpu()
    downbeat_log_prob = downbeat_log_prob.cpu()
    nonbeat_log_prob = nonbeat_log_prob.cpu()

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

    batch_size, num_frames = beat_log_prob.size()

    if threshold is None:
        offsets = torch.zeros(batch_size, dtype=torch.long)
        lengths = torch.full((batch_size,), num_frames, dtype=torch.long)
    else:
        assert 0 < threshold < 1, "Threshold should be between 0 and 1."

        beat_padding_mask = beat_log_prob < math.log(threshold)
        downbeat_padding_mask = downbeat_log_prob < math.log(threshold)
        padding_mask = beat_padding_mask & downbeat_padding_mask
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
        trimmed_downbeat_log_prob = []
        trimmed_nonbeat_log_prob = []
        lengths = []

        for _beat_log_prob, _downbeat_log_prob, _nonbeat_log_prob, offset, trimming in zip(
            beat_log_prob, downbeat_log_prob, nonbeat_log_prob, offsets, trimmings
        ):
            offset = offset.item()
            trimming = trimming.item()
            length = _beat_log_prob.size(0) - offset - trimming
            sections = [offset, num_frames - offset - trimming, trimming]
            _, _beat_log_prob, _ = torch.split(_beat_log_prob, sections, dim=-1)
            _, _downbeat_log_prob, _ = torch.split(_downbeat_log_prob, sections, dim=-1)
            _, _nonbeat_log_prob, _ = torch.split(_nonbeat_log_prob, sections, dim=-1)
            trimmed_beat_log_prob.append(_beat_log_prob)
            trimmed_downbeat_log_prob.append(_downbeat_log_prob)
            trimmed_nonbeat_log_prob.append(_nonbeat_log_prob)
            lengths.append(length)

        beat_log_prob = nn.utils.rnn.pad_sequence(
            trimmed_beat_log_prob, batch_first=True, padding_value=-float("inf")
        )
        downbeat_log_prob = nn.utils.rnn.pad_sequence(
            trimmed_downbeat_log_prob, batch_first=True, padding_value=-float("inf")
        )
        nonbeat_log_prob = nn.utils.rnn.pad_sequence(
            trimmed_nonbeat_log_prob, batch_first=True, padding_value=-float("inf")
        )
        lengths = torch.tensor(lengths, dtype=torch.long)

    peaks = _decode_beat_and_downbeat_peaks_by_viterbi(
        beat_log_prob=beat_log_prob,
        downbeat_log_prob=downbeat_log_prob,
        nonbeat_log_prob=nonbeat_log_prob,
        lengths=lengths,
        fpbs=fpbs,
        meters=meters,
        beat_region=beat_region,
        log_transition_prob=log_transition_prob,
    )

    peak_indices = []
    beats = []

    for _peaks, offset in zip(peaks, offsets):
        _peak_indices = torch.nonzero(_peaks)
        _peak_indices = _peak_indices.squeeze(dim=-1)
        _beats = _peaks[_peak_indices]

        peak_indices.append(_peak_indices + offset)
        beats.append(_beats)

    peak_indices = nn.utils.rnn.pad_sequence(peak_indices, batch_first=True, padding_value=-1)
    beats = nn.utils.rnn.pad_sequence(beats, batch_first=True, padding_value=-1)

    peak_indices = peak_indices.to(device)
    beats = beats.to(device)

    return peak_indices, beats


@torch.no_grad()
def _decode_beat_peaks_by_viterbi(
    beat_log_prob: torch.Tensor,
    nonbeat_log_prob: torch.Tensor,
    *,
    lengths: Optional[torch.Tensor] = None,
    fpbs: Optional[torch.LongTensor] = None,
    beat_region: float = 0.0625,
    log_transition_prob: Optional[float] = None,
) -> torch.LongTensor:
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
    batch_size, num_frames = beat_log_prob.size()

    if lengths is None:
        lengths = torch.full((batch_size,), num_frames, dtype=torch.long)

    if fpbs is None:
        raise ValueError("fpbs must be provided.")

    if log_transition_prob is None:
        weight = 100
        ratio = torch.abs(fpbs / fpbs.unsqueeze(dim=-1) - 1)
        log_transition_prob = F.log_softmax(-weight * ratio, dim=-1)

    peaks = torch.ops.maddad.decode_beat_peaks_by_viterbi.default(
        beat_log_prob, nonbeat_log_prob, lengths, fpbs, beat_region, log_transition_prob
    )

    return peaks


@torch.no_grad()
def _decode_beat_and_downbeat_peaks_by_viterbi(
    beat_log_prob: torch.Tensor,
    downbeat_log_prob: torch.Tensor,
    nonbeat_log_prob: torch.Tensor,
    *,
    lengths: Optional[torch.Tensor] = None,
    fpbs: Optional[torch.LongTensor] = None,
    meters: List[int] = [3, 4],
    beat_region: float = 0.0625,
    log_transition_prob: Optional[float] = None,
) -> torch.LongTensor:
    """Search for best beat path through the state space using a dynamic Bayesian network (DBN) approach.

    Args:
        logit (torch.Tensor): Logit of shape (batch_size, num_frames).
        lengths (torch.Tensor, optional): Lengths of each sample in the batch. If not provided, \
            it is assumed that all samples have the same length. Defaults to ``None``.
        fpbs (torch.Tensor): Frames per beat to assume.
        meters (list): Meters to consider. Defaults to ``[3, 4]``.
        log_transition_prob (torch.Tensor, optional): Log transition probability of shape (num_fpbs, num_fpbs). \
            If not provided, it will be computed from fpbs based on official implementation.

    Returns:
        torch.LongTensor: Section indices of shape (batch_size, num_frames).

    """
    batch_size, num_frames = beat_log_prob.size()

    if lengths is None:
        lengths = torch.full((batch_size,), num_frames, dtype=torch.long)

    if fpbs is None:
        raise ValueError("fpbs must be provided.")

    if log_transition_prob is None:
        weight = 100
        ratio = torch.abs(fpbs / fpbs.unsqueeze(dim=-1) - 1)
        log_transition_prob = F.log_softmax(-weight * ratio, dim=-1)

    peaks = []
    scores = []

    for meter in meters:
        _peaks, score = torch.ops.maddad.decode_beat_and_downbeat_peaks_by_viterbi.default(
            beat_log_prob,
            downbeat_log_prob,
            nonbeat_log_prob,
            lengths,
            fpbs,
            meter,
            beat_region,
            log_transition_prob,
        )
        peaks.append(_peaks)
        scores.append(score)

    peaks = torch.stack(peaks, dim=0)
    scores = torch.stack(scores, dim=0)
    meter_offset = torch.argmax(scores, dim=0)

    meter_offset = meter_offset.view(1, batch_size, 1)
    meter_offset = meter_offset.expand(-1, -1, num_frames)
    peaks = peaks.gather(0, meter_offset)
    peaks = peaks.squeeze(dim=0)

    return peaks
