import torch
import torch.nn as nn
import torch.nn.functional as F

from maddad.functional.dbn import (
    decode_beat_and_downbeat_peaks_by_viterbi,
    decode_beat_peaks_by_viterbi,
)


def test_decode_beat_peaks_by_viterbi() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_frames = 1000
    frame_rate = 50
    threshold = 0.8

    logit = torch.randn(batch_size, num_frames)
    beat_log_prob = F.logsigmoid(logit)
    nonbeat_log_prob = F.logsigmoid(-logit)
    peaks = decode_beat_peaks_by_viterbi(
        beat_log_prob, nonbeat_log_prob, frame_rate=frame_rate, threshold=threshold
    )

    assert peaks.dim() == 2
    assert peaks.size(0) == batch_size

    unbatched_peaks = []

    for _logit in logit:
        _beat_log_prob = F.logsigmoid(_logit)
        _nonbeat_log_prob = F.logsigmoid(-_logit)
        _peaks = decode_beat_peaks_by_viterbi(
            _beat_log_prob.unsqueeze(dim=0),
            _nonbeat_log_prob.unsqueeze(dim=0),
            frame_rate=frame_rate,
            threshold=threshold,
        )
        unbatched_peaks.append(_peaks.squeeze(dim=0))

    unbatched_peaks = nn.utils.rnn.pad_sequence(
        unbatched_peaks, batch_first=True, padding_value=-1
    )

    assert unbatched_peaks.size() == peaks.size()
    assert torch.equal(unbatched_peaks, peaks)


def test_decode_beat_and_downbeat_peaks_by_viterbi() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_frames = 1000
    frame_rate = 50
    threshold = 0.8

    beat_logit = torch.randn(batch_size, num_frames)
    downbeat_logit = torch.randn(batch_size, num_frames)
    beat_log_prob = F.logsigmoid(beat_logit)
    downbeat_log_prob = F.logsigmoid(downbeat_logit)
    nonbeat_log_prob = -torch.max(beat_log_prob, downbeat_log_prob)

    peaks, beats = decode_beat_and_downbeat_peaks_by_viterbi(
        beat_log_prob,
        downbeat_log_prob,
        nonbeat_log_prob,
        frame_rate=frame_rate,
        threshold=threshold,
    )

    assert peaks.dim() == 2
    assert peaks.size(0) == batch_size
    assert beats.dim() == 2
    assert beats.size(0) == batch_size
