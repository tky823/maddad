import os
from typing import Any, Dict, Optional, Union

import hydra
import torch
import torch.nn as nn
import torchaudio
from omegaconf import OmegaConf

from ...models.beatthis import BeatThis
from ...transforms import BeatThisTransform
from .base import BeatPipeline


class BeatThisPipeline(BeatPipeline):
    def __init__(
        self,
        model: nn.Module,
        postprocessing: Optional[str] = "dbn",
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

        self.transform = BeatThisTransform()
        self.model = model
        self.postprocessing = postprocessing
        self.chunking = chunking
        self.device = device

        self.transform.to(self.device)
        self.model.to(self.device)

        self.transform.eval()
        self.model.eval()

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
        spectrogram = self.transform(waveform)

        if self.chunking:
            output = []

            for _spectrogram in spectrogram:
                _output, _ = self.model(_spectrogram.unsqueeze(dim=0))
                output.append(_output)

            output = torch.cat(output, dim=0)
        else:
            output, _ = self.model(spectrogram)

        return output

    @classmethod
    def build_from_pretrained(
        cls, checkpoint: str, *, postprocessing: str = "dbn", device: torch.device | None = None
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

            return cls(model, postprocessing=postprocessing, device=device)
        else:
            model = BeatThis.build_from_pretrained(checkpoint)

            return cls(model, postprocessing=postprocessing, device=device)
