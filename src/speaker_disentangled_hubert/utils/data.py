import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torchaudio

from .nansy import _load_waveform, change_gender, random_eq


class LibriSpeech(torchaudio.datasets.LIBRISPEECH):
    def __init__(
        self,
        root: Union[str, Path] = "data",
        url: str = "train-clean-100",
        folder_in_archive: str = "LibriSpeech",
        download: bool = False,
        max_sample_size: Optional[int] = 80160,
    ):
        super().__init__(root, url, folder_in_archive, download)
        self.max_sample_size = max_sample_size

    def __getitem__(self, n: int) -> Dict[str, Any]:
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])

        if self.max_sample_size is not None:
            attention_mask = torch.ones(self.max_sample_size, dtype=torch.long)
            diff = len(waveform) - self.max_sample_size
            if diff > 0:
                start = random.randrange(diff)
                waveform = waveform[start : start + self.max_sample_size]
                perturbed_waveform = self.perturb_waveform(waveform, metadata[1])
            else:  # need to pad
                perturbed_waveform = self.perturb_waveform(waveform, metadata[1])
                perturbed_waveform = np.concatenate([perturbed_waveform, np.zeros(-diff, dtype=waveform.dtype)])
                waveform = np.concatenate([waveform, np.zeros(-diff, dtype=waveform.dtype)])
                attention_mask[diff:] = 0
        else:
            attention_mask = torch.ones(len(waveform), dtype=torch.long)
            perturbed_waveform = self.perturb_waveform(waveform, metadata[1])

        return {
            "waveform": torch.from_numpy(waveform),
            "perturbed_waveform": torch.from_numpy(perturbed_waveform),
            "attention_mask": attention_mask,
            "spk_id": metadata[3],
        }

    def perturb_waveform(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        perturbed_waveform = change_gender(waveform, sr)
        perturbed_waveform = random_eq(perturbed_waveform, sr)
        return np.clip(perturbed_waveform, -1.0, 1.0)


class VoxCeleb(torchaudio.datasets.VoxCeleb1Identification):
    def __init__(
        self,
        root: Union[str, Path] = "data/VoxCeleb1",
        subset: str = "train",
        meta_url: str = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt",
        download: bool = False,
        max_sample_size: Optional[int] = 128000,
    ):
        super().__init__(root, subset, meta_url, download)
        self.max_sample_size = max_sample_size

    def __getitem__(self, n: int) -> Dict[str, Any]:
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])

        if self.max_sample_size is not None:
            attention_mask = torch.ones(self.max_sample_size, dtype=torch.long)
            diff = len(waveform) - self.max_sample_size
            if diff > 0:
                start = random.randrange(diff)
                waveform = waveform[start : start + self.max_sample_size]
            else:  # need to pad
                waveform = np.concatenate([waveform, np.zeros(-diff, dtype=waveform.dtype)])
                attention_mask[diff:] = 0
        else:
            attention_mask = torch.ones(len(waveform), dtype=torch.long)

        return {
            "waveform": torch.from_numpy(waveform),
            "attention_mask": attention_mask,
            "labels": metadata[2] - 1,
        }
