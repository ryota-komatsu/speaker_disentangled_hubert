import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset, DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .utils import normalize_text


def get_collator(
    tokenizer,
):
    def collator(batch) -> Dict[str, Any]:
        inputs = ["".join(f"<{unit}>" for unit in item["units"]) for item in batch]
        inputs = tokenizer(inputs, padding=True, return_tensors="pt")
        inputs["labels"] = inputs.input_ids.masked_fill(inputs.attention_mask.bool().logical_not(), -100)

        return inputs

    return collator


def get_tokenize_fn(encoder, data_dir, text_column: str):
    data_dir = Path(data_dir).resolve()

    def _tokenize(group: pd.DataFrame):
        pos_filename = group.loc[group["correct"] == 1, "filename"].item()
        neg_filename = group.loc[group["correct"] == 0, "filename"].item()

        pos_path = str((data_dir / pos_filename).with_suffix(".wav"))
        neg_path = str((data_dir / neg_filename).with_suffix(".wav"))

        pos_audio, sr = torchaudio.load(pos_path)
        neg_audio, sr = torchaudio.load(neg_path)

        input_values = [pos_audio.squeeze(0), neg_audio.squeeze(0)]
        attention_mask = [torch.ones_like(item, dtype=torch.long) for item in input_values]

        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)

        outputs = encoder(input_values.to(encoder.device), attention_mask.to(encoder.device))

        example = {
            "filename": {
                "pos": pos_filename,
                "neg": neg_filename,
            },
            "text": {
                "pos": group.loc[group["correct"] == 1, text_column].item(),
                "neg": group.loc[group["correct"] == 0, text_column].item(),
            },
            "units": {
                "pos": outputs[0]["units"].tolist(),
                "neg": outputs[1]["units"].tolist(),
            },
        }

        if "frequency" in group.columns:
            example["frequency"] = group.loc[group["correct"] == 1, "frequency"].item()

        return pd.Series(example)

    return _tokenize


def tokenize_eval(config):
    from ..s5hubert import S5HubertForSyllableDiscovery

    tqdm.pandas()

    app_dir = Path(config.dataset.APP_DIR).expanduser()
    tSC_dir = Path(config.dataset.tSC_DIR)

    swuggy_dev_dir = app_dir / "datasets/sLM21-dataset/lexical/dev"
    sblimp_dev_dir = app_dir / "datasets/sLM21-dataset/syntactic/dev"
    swuggy_test_dir = app_dir / "datasets/sLM21-dataset/lexical/test"
    sblimp_test_dir = app_dir / "datasets/sLM21-dataset/syntactic/test"

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")

    # sWUGGY
    swuggy_dev = pd.read_csv(swuggy_dev_dir / "gold.csv")
    swuggy_dev = swuggy_dev.groupby(["id", "voice"])
    swuggy_dev = swuggy_dev.progress_apply(get_tokenize_fn(encoder, swuggy_dev_dir, "word"), include_groups=False)
    swuggy_dev = Dataset.from_pandas(swuggy_dev)

    swuggy_test = pd.read_csv(swuggy_test_dir / "gold.csv")
    swuggy_test = swuggy_test.groupby(["id", "voice"])
    swuggy_test = swuggy_test.progress_apply(get_tokenize_fn(encoder, swuggy_test_dir, "word"), include_groups=False)
    swuggy_test = Dataset.from_pandas(swuggy_test)

    # sBLIMP
    sblimp_dev = pd.read_csv(sblimp_dev_dir / "gold.csv")
    sblimp_dev = sblimp_dev.groupby(["id", "voice", "subtype"])
    sblimp_dev = sblimp_dev.progress_apply(
        get_tokenize_fn(encoder, sblimp_dev_dir, "transcription"), include_groups=False
    )
    sblimp_dev = Dataset.from_pandas(sblimp_dev)

    sblimp_test = pd.read_csv(sblimp_test_dir / "gold.csv")
    sblimp_test = sblimp_test.groupby(["id", "voice", "subtype"])
    sblimp_test = sblimp_test.progress_apply(
        get_tokenize_fn(encoder, sblimp_test_dir, "transcription"), include_groups=False
    )
    sblimp_test = Dataset.from_pandas(sblimp_test)

    # tSC
    tSC_test_paths = sorted(glob.glob(os.path.join(tSC_dir, "*.wav")), key=lambda x: int(Path(x).stem.split("_")[0]))
    tSC_test = []

    for n in tqdm(range(0, len(tSC_test_paths), 2)):
        pos_path = tSC_test_paths[n]
        neg_path = tSC_test_paths[n + 1]

        pos_audio, sr = torchaudio.load(pos_path)
        neg_audio, sr = torchaudio.load(neg_path)

        input_values = [pos_audio.squeeze(0), neg_audio.squeeze(0)]
        attention_mask = [torch.ones_like(item, dtype=torch.long) for item in input_values]

        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)

        outputs = encoder(input_values.to(encoder.device), attention_mask.to(encoder.device))

        example = {
            "filename": {
                "pos": pos_path,
                "neg": neg_path,
            },
            "units": {
                "pos": outputs[0]["units"].tolist(),
                "neg": outputs[1]["units"].tolist(),
            },
        }
        tSC_test.append(example)

    tSC_test = Dataset.from_list(tSC_test)

    swuggy = DatasetDict({"validation": swuggy_dev, "test": swuggy_test})
    sblimp = DatasetDict({"validation": sblimp_dev, "test": sblimp_test})
    tSC = DatasetDict({"test": tSC_test})

    swuggy.push_to_hub(config.dataset.name, "sWUGGY")
    sblimp.push_to_hub(config.dataset.name, "sBLIMP")
    tSC.push_to_hub(config.dataset.name, "tSC")


def tokenize_train(config, num_shards: int = 1, shard_index: int = 0):
    from ..s5hubert import S5HubertForSyllableDiscovery

    dataset = load_dataset(config.dataset.train, split="train")
    dataset = dataset.shard(num_shards=num_shards, index=shard_index)

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")

    with open(f"{config.dataset.manifest_prefix}{shard_index}.json", "w") as f:
        for example in tqdm(dataset):
            load_path = os.path.join(config.dataset.ll_dir, example["recording"]["id"] + config.dataset.ext_audio)
            save_path = os.path.join(config.dataset.lh_dir, example["id"] + config.dataset.ext_audio)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            input_values, sr = torchaudio.load(
                load_path,
                frame_offset=math.floor(16000 * max(example["start"], 0)),
                num_frames=math.floor(16000 * example["duration"]),
            )
            torchaudio.save(save_path, input_values, sr, encoding="PCM_S", bits_per_sample=16)

            outputs = encoder(input_values.to(encoder.device))

            text = example["supervisions"][0]["custom"]["texts"][0]
            text = normalize_text(text)

            manifest = {
                "audio_filepath": save_path,
                "text": text,
                "id": example["id"],
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
            }
            f.write(json.dumps(manifest) + "\n")


def align_train(config, num_shards: int = 1):
    data_files = [f"{config.dataset.manifest_prefix}{shard_index}.json" for shard_index in range(num_shards)]
    dataset = load_dataset("json", data_files=data_files, split="train")

    id_to_aligned_text = dict()

    for shard_index in range(num_shards):
        manifest_prefix = Path(config.dataset.manifest_prefix).stem
        manifest_with_output_file_paths = os.path.join(
            config.dataset.lh_dir, f"shard{shard_index}/{manifest_prefix}{shard_index}_with_output_file_paths.json"
        )

        with open(manifest_with_output_file_paths) as f:
            for example in f:
                example = json.loads(example.strip())

                id_ = str(Path(example["audio_filepath"]).relative_to(config.dataset.lh_dir).with_suffix(""))
                aligned_text = []

                if "words_level_ctm_filepath" in example:
                    with open(example["words_level_ctm_filepath"]) as g:
                        for line in g:
                            _, _, start, duration, word, _, _, _ = line.split(" ")

                            aligned_text.append(
                                {
                                    "start_time": float(start),
                                    "end_time": float(start) + float(duration),
                                    "word": " " + word,
                                }
                            )

                id_to_aligned_text[id_] = aligned_text

    def add_aligned_units(example):
        example["aligned_text"] = id_to_aligned_text[example["id"]]

        if not example["aligned_text"]:
            end_time = sum(example["durations"]) * 0.02
            example["aligned_units"] = [
                {"start_time": 0.0, "end_time": end_time, "units": example["units"], "text": example["text"]}
            ]
            return example

        unit_timestamps = np.cumsum(example["durations"]) * 0.02
        word_timestamps = sorted(
            {item["start_time"] for item in example["aligned_text"]}
            | {item["end_time"] for item in example["aligned_text"]}
        )
        aligned_timestamps = sorted(
            set(unit_timestamps) & set(word_timestamps) | set([max(unit_timestamps[-1], word_timestamps[-1])])
        )

        aligned_units = []
        start_time = 0

        for end_time in aligned_timestamps:
            units = [
                unit
                for unit, unit_end_time in zip(example["units"], unit_timestamps)
                if start_time < unit_end_time <= end_time
            ]
            text = "".join(
                item["word"]
                for item in example["aligned_text"]
                if start_time <= item["start_time"] and item["end_time"] <= end_time
            )

            aligned_units.append({"start_time": start_time, "end_time": end_time, "units": units, "text": text})
            start_time = end_time

        example["aligned_units"] = aligned_units

        return example

    dataset = dataset.map(add_aligned_units, remove_columns="audio_filepath")
    dataset = DatasetDict({"train": dataset})
    dataset.push_to_hub(config.dataset.name, "Libri-Light")
