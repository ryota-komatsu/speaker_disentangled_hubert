import glob
import os
from pathlib import Path
from typing import Any, Dict

import librosa
import torch
from datasets import Array2D, Audio, Features, Sequence, Value, load_dataset
from torch.nn.utils.rnn import pad_sequence

from ..bigvgan.data import mel_spectrogram
from ..s5hubert import S5HubertForSyllableDiscovery


def get_collate_fn(pad_token_id: int = 16384):
    def collate_fn(batch) -> Dict[str, Any]:
        input_ids = [item["units"] for item in batch]
        spectrogram_labels = [item["spectrogram"] for item in batch]
        duration_labels = [item["durations"] for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        spectrogram_labels = pad_sequence(spectrogram_labels, batch_first=True, padding_value=-100)
        duration_labels = pad_sequence(duration_labels, batch_first=True)

        return {
            "input_ids": input_ids,
            "spectrogram_labels": spectrogram_labels,
            "duration_labels": duration_labels,
        }

    return collate_fn


def tokenize(config):
    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")

    features = Features(
        {
            "audio": Audio(sampling_rate=16000),
            "id": Value("string"),
            "units": Sequence(Value("int32")),
            "durations": Sequence(Value("int32")),
            "transcript": Value("string"),
            "spectrogram": Array2D(shape=(None, 80), dtype="float32"),
        }
    )

    # LibriTTS-R
    data_files = {
        "train": glob.glob(os.path.join(config.dataset.libritts_dir, "train-*/**/*.wav"), recursive=True),
        "dev": glob.glob(os.path.join(config.dataset.libritts_dir, "dev-clean/**/*.wav"), recursive=True),
    }
    dataset = load_dataset("audiofolder", data_files=data_files, features=features)
    dataset = dataset.map(
        get_tokenize_fn(encoder, config.dataset.libritts_dir, ".normalized.txt"), remove_columns="audio"
    )
    dataset.push_to_hub(config.dataset.name, "LibriTTS-R")

    # Hi-Fi-CAPTAIN
    data_files = {
        "female": glob.glob(os.path.join(config.dataset.hfc_dir, "female/**/*.wav"), recursive=True),
    }
    dataset = load_dataset("audiofolder", data_files=data_files, features=features)
    dataset = dataset.map(get_tokenize_fn(encoder, config.dataset.hfc_dir, ""), remove_columns="audio")
    dataset.push_to_hub(config.dataset.name, "Hi-Fi-CAPTAIN")


def get_tokenize_fn(encoder, data_dir, ext_txt: str = ".normalized.txt"):
    data_dir = Path(data_dir).resolve()

    def _tokenize(example):
        input_values = example["audio"]["array"]
        input_values = librosa.effects.trim(input_values, top_db=20)[0]
        input_values = torch.from_numpy(input_values)
        input_values = input_values.to(encoder.device, torch.float)
        input_values = input_values / input_values.abs().max() * 0.95
        input_values = input_values.unsqueeze(0)

        spectrogram_labels = mel_spectrogram(input_values).squeeze(0)  # (80, len)
        spectrogram_labels = spectrogram_labels.transpose(0, 1)  # (len, 80)
        spectrogram_labels = spectrogram_labels.cpu().tolist()

        outputs = encoder(input_values)

        id = str(Path(example["audio"]["path"]).relative_to(data_dir).with_suffix(""))
        txt_path = Path(example["audio"]["path"]).with_suffix(ext_txt)

        transcript = ""
        if txt_path.is_file():
            with open(txt_path) as g:
                transcript = g.read().rstrip()

        return {
            "id": id,
            "units": outputs[0]["units"].tolist(),
            "durations": outputs[0]["durations"].tolist(),
            "transcript": transcript,
            "spectrogram": spectrogram_labels,
        }

    return _tokenize
