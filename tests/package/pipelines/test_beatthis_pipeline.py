import os

import pytest
import torch
import torchaudio
from maddad_testing.utils import maddad_testing_cache_dir as cache_dir

from maddad.pipelines.beat import BeatThisPipeline
from maddad.utils._github import download_file_from_github_release


def test_beatthis_pipeline(audio_path: str) -> None:
    waveform, sample_rate = torchaudio.load(audio_path)

    assert sample_rate == 22050

    pipeline = BeatThisPipeline.build_from_pretrained("official_beatthis", device="cpu")

    waveform = waveform.mean(dim=0)
    output = pipeline(waveform, sample_rate=sample_rate)
    beat = output["beat"]

    url = "https://github.com/tky823/maddad/releases/download/v0.0.0/test_official_beatthis.pth"
    path = os.path.join(cache_dir, "test_official_beatthis.pth")
    download_file_from_github_release(url, path)
    data = torch.load(path)
    expected_beat = data["beat"]

    assert beat.size() == expected_beat.size()

    torch.allclose(beat, expected_beat, atol=1e-6)


def test_beatthis_pipeline_with_dbn(audio_path: str) -> None:
    # TODO: regression test
    waveform, sample_rate = torchaudio.load(audio_path)

    assert sample_rate == 22050

    pipeline = BeatThisPipeline.build_from_pretrained(
        "official_beatthis",
        decoder="dbn",
    )

    waveform = waveform.mean(dim=0)
    output = pipeline(waveform, sample_rate=sample_rate)

    assert set(output.keys()) == {"beat"}

    beat = output["beat"]

    assert beat.dim() == 1


@pytest.fixture
def audio_path() -> str:
    url = "https://github.com/tky823/maddad/releases/download/v0.0.0/little-fugue.mp3"
    path = os.path.join(cache_dir, "little-fugue.mp3")
    download_file_from_github_release(url, path)

    return path
