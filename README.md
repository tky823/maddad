# maddad

Toolkit for beat detection.

## Installation

You can install by pip.

```shell
pip install git+https://github.com/tky823/maddad.git
```

or clone this repository.

```shell
git clone https://github.com/tky823/maddad.git
cd maddad
pip install -e .
```

## Usage

From file:

```python
>>> from maddad.pipelines.beat_and_downbeat import BeatThisPipeline
>>> audio_path = "audio.mp3"
>>> decoder = "minimal"  # "minimal" or "dbn"
>>> pipeline = BeatThisPipeline.build_from_pretrained("official_beatthis", decoder=decoder, device="cpu")
>>> output = pipeline(audio_path)
>>> output["beat"]
tensor([ 0.0600,  0.8400, ..., 38.2000])
>>> output["downbeat"]
tensor([ 0.0600,  0.8400, ..., 38.2000])
```

From tensor:

```python
>>> import torchaudio
>>> from maddad.pipelines.beat_and_downbeat import BeatThisPipeline
>>> audio_path = "audio.mp3"
>>> decoder = "minimal"  # "minimal" or "dbn"
>>> waveform, sample_rate = torchaudio.load(audio_path)
>>> waveform = waveform.mean(dim=0)  # Channel dimension should be removed.
>>> pipeline = BeatThisPipeline.build_from_pretrained("official_beatthis", decoder=decoder, device="cpu")
>>> output = pipeline(waveform, sample_rate=sample_rate)
>>> output["beat"]
tensor([ 0.0600,  0.8400, ..., 38.2000])
>>> output["downbeat"]
tensor([ 0.0600,  0.8400, ..., 38.2000])
```

## License

- CC BY-NC 4.0
