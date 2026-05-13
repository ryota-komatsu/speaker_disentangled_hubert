import json
from pathlib import Path

import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm import tqdm

from ...s5hubert import SylRegForSyllableDiscovery


def tokenize_clean(
    num_shards: int = 1,
    shard_index: int = 0,
    data_dir: str = "data/peoples_speech",
    model_name_or_path: str = "ryota-komatsu/SylReg-Distill",
):
    dataset = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
    dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.with_format("torch")

    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / f"manifest_clean{shard_index}.json"

    with open(manifest_path, "w") as f:
        for example in tqdm(dataset):
            # if filter_fn(example):
            #     continue

            id_ = str((Path("clean/train") / example["id"]).with_suffix(""))
            audio_filepath = (Path(data_dir) / id_).with_suffix(".flac")
            audio_filepath.parent.mkdir(parents=True, exist_ok=True)
            audio_filepath = str(audio_filepath)

            input_values = example["audio"]["array"].unsqueeze(0)

            outputs = encoder(input_values.to(encoder.device))

            example = {
                "id": id_,
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
            }
            json.dump(example, f)
            f.write("\n")


def tokenize_clean_sa(
    data_dir: str = "data/peoples_speech",
    model_name_or_path: str = "ryota-komatsu/SylReg-Distill",
):
    dataset = load_dataset("MLCommons/peoples_speech", "clean_sa", split="train", streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.with_format("torch")

    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / "manifest_clean_sa.json"

    with open(manifest_path, "w") as f:
        for example in tqdm(dataset):
            # if filter_fn(example):
            #     continue

            id_ = str((Path("clean_sa/train") / example["id"]).with_suffix(""))
            audio_filepath = (Path(data_dir) / id_).with_suffix(".flac")
            audio_filepath.parent.mkdir(parents=True, exist_ok=True)
            audio_filepath = str(audio_filepath)

            input_values = example["audio"]["array"].unsqueeze(0)

            outputs = encoder(input_values.to(encoder.device))

            example = {
                "id": id_,
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
            }
            json.dump(example, f)
            f.write("\n")
