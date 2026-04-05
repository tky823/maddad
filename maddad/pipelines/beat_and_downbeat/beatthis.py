import os
import warnings
from typing import Any, Dict, Optional, Union

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf

from ...models.beatthis import BeatThis
from ...transforms import BeatThisTransform
from ...transforms.beat_and_downbeat import DBNBeatAndDownbeatDecoder
from ...transforms.beatthis import MinimalBeatAndDownbeatDecoder
from .base import BeatAndDownbeatPipeline


class BeatThisPipeline(BeatAndDownbeatPipeline):
    def __init__(
        self,
        model: nn.Module,
        transform: Optional[nn.Module] = None,
        decoder: Optional[Union[nn.Module, str]] = None,
        chunking: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        if transform is None:
            transform = BeatThisTransform()
        elif not isinstance(transform, BeatThisTransform):
            warnings.warn(
                f"{type(transform)} is not supported, which may cause unexpected behavior.",
                UserWarning,
            )

        if decoder is None:
            decoder = MinimalBeatAndDownbeatDecoder()
        elif isinstance(decoder, str):
            if decoder == "minimal":
                decoder = MinimalBeatAndDownbeatDecoder()
            elif decoder == "dbn":
                frame_rate = int(transform.sample_rate / transform.hop_length)
                decoder = DBNBeatAndDownbeatDecoder(frame_rate=frame_rate)
            else:
                raise ValueError(f"Unsupported decoder {decoder}.")
        else:
            raise ValueError(f"Unsupported decoder type {type(decoder)}.")

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.transform = transform
        self.model = model
        self.decoder = decoder
        self.chunking = chunking
        self.device = device

        self.transform.to(self.device)
        self.model.to(self.device)
        self.decoder.to(self.device)

        self.transform.eval()
        self.model.eval()
        self.decoder.eval()

    def __call__(
        self, input: Union[torch.Tensor, str], sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        if isinstance(input, str):
            waveform, _sample_rate = torchaudio.load(input)

            if sample_rate is not None:
                assert _sample_rate == sample_rate, (
                    f"Expected sample rate {sample_rate}, but got {_sample_rate}."
                )

            waveform = waveform.mean(dim=0)  # Channel dimension should be removed.
            sample_rate = _sample_rate
        else:
            waveform = input

            if sample_rate is None:
                raise ValueError("Sample rate must be provided for tensor input.")

            assert waveform.dim() == 1, (
                f"Shape of waveform should be (num_samples,), but got {waveform.size()}."
            )

        return self.forward(waveform, sample_rate=sample_rate)

    @torch.inference_mode()
    def forward(self, waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        assert waveform.dim() == 1, f"Expected mono waveform, but got {waveform.dim()} channels."

        if self.transform.sample_rate != sample_rate:
            raise ValueError(
                f"Expected sample rate {self.transform.sample_rate}, but got {sample_rate}."
            )

        waveform = waveform.to(self.device)

        if isinstance(self.transform, BeatThisTransform):
            spectrogram, last_offset = self.transform(waveform)
        else:
            spectrogram = self.transform(waveform)
            last_offset = None

        if self.chunking:
            beat_logit = []
            downbeat_logit = []

            for _spectrogram in spectrogram:
                _beat_logit, _downbeat_logit = self.model(_spectrogram.unsqueeze(dim=0))
                beat_logit.append(_beat_logit)
                downbeat_logit.append(_downbeat_logit)

            beat_logit = torch.cat(beat_logit, dim=0)
            downbeat_logit = torch.cat(downbeat_logit, dim=0)
        else:
            beat_logit, downbeat_logit = self.model(spectrogram)

        if isinstance(self.transform, BeatThisTransform):
            pad = self.transform.pad
            beat_logit = F.pad(beat_logit, (-pad, -pad))
            downbeat_logit = F.pad(downbeat_logit, (-pad, -pad))
            num_chunks = beat_logit.size(0)

            beat_logit, last_beat_logit = torch.split(beat_logit, [num_chunks - 1, 1], dim=0)
            _, last_beat_logit = torch.split(
                last_beat_logit, [last_offset, last_beat_logit.size(-1) - last_offset], dim=-1
            )
            downbeat_logit, last_downbeat_logit = torch.split(
                downbeat_logit, [num_chunks - 1, 1], dim=0
            )
            _, last_downbeat_logit = torch.split(
                last_downbeat_logit,
                [last_offset, last_downbeat_logit.size(-1) - last_offset],
                dim=-1,
            )
            beat_logit = beat_logit.view(1, -1)
            beat_logit = torch.cat([beat_logit, last_beat_logit], dim=-1)
            downbeat_logit = downbeat_logit.view(1, -1)
            downbeat_logit = torch.cat([downbeat_logit, last_downbeat_logit], dim=-1)
        else:
            warnings.warn(
                f"{type(self.transform)} is not supported, so no trimming is applied.",
                UserWarning,
            )

        beat_peaks, indices = self.decoder(beat_logit, downbeat_logit)
        beat_peaks = beat_peaks.squeeze(dim=0)
        indices = indices.squeeze(dim=0)
        downbeat_peaks = beat_peaks[indices == 1]

        frame_rate = self.transform.sample_rate / self.transform.hop_length

        output = {
            "beat": beat_peaks / frame_rate,
            "downbeat": downbeat_peaks / frame_rate,
        }

        return output

    @classmethod
    def build_from_pretrained(
        cls,
        checkpoint: str,
        *,
        decoder: Optional[Union[nn.Module, str]] = None,
        device: torch.device | None = None,
    ) -> "BeatThisPipeline":
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        if os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            resolved_config = OmegaConf.load(state_dict["config"])
            model: BeatThis = hydra.utils.instantiate(resolved_config.model)
            model_state_dict = state_dict["model"]
            model.load_state_dict(model_state_dict)

            return cls(model, decoder=decoder, device=device)
        else:
            model = BeatThis.build_from_pretrained(checkpoint)

            return cls(model, decoder=decoder, device=device)
