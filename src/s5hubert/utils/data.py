import os
import random
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from .nansy import _load_waveform, change_gender, random_eq


def shard(dataset, num_shards: int, index: int):
    div, mod = divmod(len(dataset), num_shards)
    start = div * index + min(index, mod)
    end = start + div + (1 if index < mod else 0)
    indices = range(start, end)
    return torch.utils.data.Subset(dataset, indices)


class LibriSpeech(torchaudio.datasets.LIBRISPEECH):
    def __init__(
        self,
        root: str | Path = "data",
        url: str = "train-clean-100",
        folder_in_archive: str = "LibriSpeech",
        download: bool = False,
        max_sample_size: int | None = 80080,
        perturb: bool = True,
    ):
        super().__init__(root, url, folder_in_archive, download)
        self.max_sample_size = max_sample_size
        self.perturb = perturb

    def __getitem__(self, n: int) -> Dict[str, Any]:
        metadata = self.get_metadata(n)
        teacher_input_values = _load_waveform(os.path.join(self._archive, metadata[0]))

        if self.max_sample_size is not None:
            teacher_input_values, _ = librosa.effects.trim(teacher_input_values, top_db=20)
            diff = len(teacher_input_values) - self.max_sample_size
            if diff > 0:
                start = random.randrange(diff)
                teacher_input_values = teacher_input_values[start : start + self.max_sample_size]

        student_input_values = (
            self.perturb_waveform(teacher_input_values, metadata[1]) if self.perturb else teacher_input_values
        )

        return {
            "teacher_input_values": torch.from_numpy(teacher_input_values),
            "student_input_values": torch.from_numpy(student_input_values),
            "spk_id": metadata[3],
            "wav_name": metadata[0],
        }

    def perturb_waveform(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        student_input_values = change_gender(waveform, sr)
        student_input_values = random_eq(student_input_values, sr)
        return np.clip(student_input_values, -1.0, 1.0)

    @staticmethod
    def collate_fn(batch) -> Dict[str, torch.Tensor]:
        teacher_input_values = [item["teacher_input_values"] for item in batch]
        student_input_values = [item["student_input_values"] for item in batch]

        teacher_attention_mask = [torch.ones_like(item["teacher_input_values"], dtype=torch.long) for item in batch]
        student_attention_mask = [torch.ones_like(item["student_input_values"], dtype=torch.long) for item in batch]

        teacher_input_values = pad_sequence(teacher_input_values, batch_first=True)
        student_input_values = pad_sequence(student_input_values, batch_first=True)
        teacher_attention_mask = pad_sequence(teacher_attention_mask, batch_first=True)
        student_attention_mask = pad_sequence(student_attention_mask, batch_first=True)

        wav_names = [item["wav_name"] for item in batch]

        return {
            "teacher_input_values": teacher_input_values,
            "student_input_values": student_input_values,
            "teacher_attention_mask": teacher_attention_mask,
            "student_attention_mask": student_attention_mask,
            "wav_names": wav_names,
        }

    @staticmethod
    def collate_fn2(batch) -> Dict[str, torch.Tensor]:
        teacher_input_values = [item["teacher_input_values"].unsqueeze(0) for item in batch]
        student_input_values = [item["student_input_values"].unsqueeze(0) for item in batch]
        return {
            "teacher_input_values": teacher_input_values,
            "student_input_values": student_input_values,
        }


class LibriLight(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        max_sample_size: int | None = 80080,
        perturb: bool = True,
    ):
        self.max_sample_size = max_sample_size
        self.perturb = perturb
        self.paths = list(Path(data_dir).glob("**/*.flac"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        teacher_input_values = _load_waveform(self.paths[n])

        if self.max_sample_size is not None:
            teacher_input_values, _ = librosa.effects.trim(teacher_input_values, top_db=20)
            diff = len(teacher_input_values) - self.max_sample_size
            if diff > 0:
                start = random.randrange(diff)
                teacher_input_values = teacher_input_values[start : start + self.max_sample_size]

        student_input_values = self.perturb_waveform(teacher_input_values) if self.perturb else teacher_input_values

        return {
            "teacher_input_values": torch.from_numpy(teacher_input_values),
            "student_input_values": torch.from_numpy(student_input_values),
        }

    def perturb_waveform(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        student_input_values = change_gender(waveform, sr)
        student_input_values = random_eq(student_input_values, sr)
        return np.clip(student_input_values, -1.0, 1.0)

    @staticmethod
    def collate_fn(batch) -> Dict[str, torch.Tensor]:
        teacher_input_values = [item["teacher_input_values"] for item in batch]
        student_input_values = [item["student_input_values"] for item in batch]

        teacher_attention_mask = [torch.ones_like(item["teacher_input_values"], dtype=torch.long) for item in batch]
        student_attention_mask = [torch.ones_like(item["student_input_values"], dtype=torch.long) for item in batch]

        teacher_input_values = pad_sequence(teacher_input_values, batch_first=True)
        student_input_values = pad_sequence(student_input_values, batch_first=True)
        teacher_attention_mask = pad_sequence(teacher_attention_mask, batch_first=True)
        student_attention_mask = pad_sequence(student_attention_mask, batch_first=True)

        return {
            "teacher_input_values": teacher_input_values,
            "student_input_values": student_input_values,
            "teacher_attention_mask": teacher_attention_mask,
            "student_attention_mask": student_attention_mask,
        }
