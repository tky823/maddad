import torch

from maddad.functional.dbn import decode_beat_peaks_by_viterbi


def test_decode_beat_peaks_by_viterbi() -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_frames = 2000
    frame_rate = 50

    logit = torch.randn(batch_size, num_frames)
    peaks = decode_beat_peaks_by_viterbi(logit, frame_rate=frame_rate)

    assert peaks.dim() == 2
    assert peaks.size(0) == batch_size
