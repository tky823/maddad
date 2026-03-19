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
from ...transforms.beatthis import MinimalBeatDecoder
from .base import BeatPipeline


class BeatThisPipeline(BeatPipeline):
    def __init__(
        self,
        model: nn.Module,
        decoder: Optional[nn.Module] = None,
        chunking: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        if decoder is None:
            decoder = MinimalBeatDecoder()

        self.transform = BeatThisTransform()
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

            sample_rate = _sample_rate
        else:
            waveform = input

            if sample_rate is None:
                raise ValueError("Sample rate must be provided for tensor input.")

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
            logit = []

            for _spectrogram in spectrogram:
                output, _ = self.model(_spectrogram.unsqueeze(dim=0))
                logit.append(output)

            logit = torch.cat(logit, dim=0)
        else:
            logit, _ = self.model(spectrogram)

        if isinstance(self.transform, BeatThisTransform):
            pad = self.transform.pad
            logit = F.pad(logit, (-pad, -pad))
            num_chunks = logit.size(0)

            logit, last_logit = torch.split(logit, [num_chunks - 1, 1], dim=0)
            _, last_logit = torch.split(
                last_logit, [last_offset, last_logit.size(-1) - last_offset], dim=-1
            )
            logit = logit.view(1, -1)
            logit = torch.cat([logit, last_logit], dim=-1)
        else:
            warnings.warn(
                f"{type(self.transform)} is not supported, so no trimming is applied.",
                UserWarning,
            )

        peaks = self.decoder(logit)
        peaks = peaks.squeeze(dim=0)

        frame_rate = self.transform.hop_length / self.transform.sample_rate

        output = {
            "beat": peaks * frame_rate,
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
