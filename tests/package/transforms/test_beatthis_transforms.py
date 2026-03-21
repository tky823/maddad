import os

import torch
from maddad_testing.utils import maddad_testing_cache_dir as cache_dir

from maddad.transforms import BeatThisMelSpectrogram
from maddad.utils._github import download_file_from_github_release


def test_beatthis_melspectrogram() -> None:
    url = "https://github.com/tky823/maddad/releases/download/v0.0.0/test_beatthis_melspectrogram.pth"
    path = os.path.join(cache_dir, "test_beatthis_melspectrogram.pth")
    download_file_from_github_release(url, path)

    data = torch.load(path)
    waveform = data["waveform"]
    sample_rate = data["sample_rate"]
    expected_spectrogram = data["melspectrogram"]

    assert sample_rate == 22050

    transform = BeatThisMelSpectrogram()

    spectrogram = transform(waveform)

    assert torch.allclose(spectrogram, expected_spectrogram, atol=1e-4)
