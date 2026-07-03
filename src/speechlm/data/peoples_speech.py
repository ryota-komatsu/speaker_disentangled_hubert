import json
from pathlib import Path

from better_profanity import profanity
from datasets import Audio, load_dataset
from tqdm import tqdm

from ...s5hubert import SylRegForSyllableDiscovery
from .utils import (
    add_aligned_units,
    filler_pattern1,
    filler_pattern2,
    get_aligner,
    repeat_pattern1,
    repeat_pattern2,
    single_word_pattern,
)


def filter_fn(example: dict):
    return (
        profanity.contains_profanity(example["text"])
        or filler_pattern1.search(example["text"])
        or filler_pattern2.search(example["text"])
        or repeat_pattern1.search(example["text"])
        or repeat_pattern2.search(example["text"])
        or single_word_pattern.search(example["text"])
    )


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
    aligner = get_aligner()

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / f"manifest_clean{shard_index}.json"

    with open(manifest_path, "w") as f:
        for example in tqdm(dataset):
            if filter_fn(example):
                continue

            id_ = str((Path("clean/train") / example["id"]).with_suffix(""))

            input_values = example["audio"]["array"].unsqueeze(0)

            outputs = encoder(input_values.to(encoder.device))

            example = {
                "id": id_,
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
                "aligned_text": aligner(input_values, example["text"]),
            }
            example = add_aligned_units(example)
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
    aligner = get_aligner()

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / "manifest_clean_sa.json"

    with open(manifest_path, "w") as f:
        for example in tqdm(dataset):
            if filter_fn(example):
                continue

            id_ = str((Path("clean_sa/train") / example["id"]).with_suffix(""))

            input_values = example["audio"]["array"].unsqueeze(0)

            outputs = encoder(input_values.to(encoder.device))

            example = {
                "id": id_,
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
                "aligned_text": aligner(input_values, example["text"]),
            }
            example = add_aligned_units(example)
            json.dump(example, f)
            f.write("\n")
