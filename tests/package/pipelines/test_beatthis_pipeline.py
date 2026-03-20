import os

import torchaudio
from maddad_testing.utils import maddad_testing_cache_dir as cache_dir

from maddad.pipelines.beat import BeatThisPipeline
from maddad.utils._github import download_file_from_github_release


def test_beatthis_pipeline() -> None:
    url = "https://github.com/tky823/maddad/releases/download/v0.0.0/little-fugue.mp3"
    path = os.path.join(cache_dir, "little-fugue.mp3")
    download_file_from_github_release(url, path)

    waveform, sample_rate = torchaudio.load(path)

    assert sample_rate == 22050

    pipeline = BeatThisPipeline.build_from_pretrained("official_beatthis", device="cpu")

    waveform = waveform.mean(dim=0)
    _ = pipeline(waveform, sample_rate=sample_rate)
