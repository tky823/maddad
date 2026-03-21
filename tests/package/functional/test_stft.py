import os

import torch
from maddad_testing.utils import maddad_testing_cache_dir as cache_dir

from maddad.functional import stft
from maddad.utils._github import download_file_from_github_release


def test_stft() -> None:
    url = "https://github.com/tky823/maddad/releases/download/v0.0.0/test_madmom_stft.pth"
    path = os.path.join(cache_dir, "test_madmom_stft.pth")
    download_file_from_github_release(url, path)

    data = torch.load(path)
    waveform = data["waveform"]
    sample_rate = data["sample_rate"]
    expected_output = data["spectrogram"]

    frame_size = 2048
    hop_size = 441

    assert sample_rate == 22050

    output = stft(waveform, n_fft=frame_size, hop_length=hop_size)

    assert torch.allclose(output, expected_output, atol=1e-4)
