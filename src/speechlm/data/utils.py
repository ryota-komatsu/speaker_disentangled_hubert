import glob
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoProcessor
from transformers.models.qwen3_asr.processing_qwen3_asr import _is_cjk_char, _is_kept_char

from ...s5hubert import SylRegForSyllableDiscovery

filler_pattern1 = re.compile(r"\buhm?,?\b", re.IGNORECASE)
filler_pattern2 = re.compile(r"\bum,?\b", re.IGNORECASE)
repeat_pattern1 = re.compile(r"\b(\w+)\b([,\s]+\1\b)+", re.IGNORECASE)
repeat_pattern2 = re.compile(r"\b(\w+\s+\w+)\b([,\s]+\1\b)+", re.IGNORECASE)
single_word_pattern = re.compile(r"^\w+$", re.IGNORECASE)


def get_collator(tokenizer, max_length: int = 128, speech_segment_prob: float = 0.4):
    def collator(batch) -> dict[str, Any]:
        inputs = []
        for item in batch:
            # text-only
            if item["text"]:
                item = item["text"]

            # speech-only
            elif not item["aligned_units"]:
                item = "".join(f"<{unit}>" for unit in item["units"])

            # speech-text interleaving
            else:
                is_speech = np.random.rand(len(item["aligned_units"])) < speech_segment_prob
                is_speech[1:] &= ~is_speech[:-1]  # p(1-p) = 0.4 * 0.6 = 0.24

                item = "".join(
                    "".join(f"<{unit}>" for unit in chunk["units"]) if is_speech_chunk else chunk["text"]
                    for is_speech_chunk, chunk in zip(is_speech, item["aligned_units"])
                ).lstrip()

            inputs.append(item + tokenizer.eos_token)

        inputs = tokenizer(inputs, padding=False)

        # random truncation
        lengths = torch.tensor([len(input_ids) for input_ids in inputs.input_ids])
        starts = (torch.rand(len(lengths)) * torch.clamp(lengths - max_length, min=0)).int()
        input_ids = [inputs.input_ids[i][start : start + max_length] for i, start in enumerate(starts)]

        inputs = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
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

        pos_outputs = encoder(pos_audio.to(encoder.device))
        neg_outputs = encoder(neg_audio.to(encoder.device))

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
                "pos": pos_outputs[0]["units"].tolist(),
                "neg": neg_outputs[0]["units"].tolist(),
            },
        }

        if "frequency" in group.columns:
            example["frequency"] = group.loc[group["correct"] == 1, "frequency"].item()

        return pd.Series(example)

    return _tokenize


def tokenize_storycloze(encoder, SC_dir):
    SC_test_paths = sorted(glob.glob(os.path.join(SC_dir, "*.wav")), key=lambda x: int(Path(x).stem.split("_")[0]))
    SC_test = []

    for n in tqdm(range(0, len(SC_test_paths), 2)):
        pos_path = SC_test_paths[n]
        neg_path = SC_test_paths[n + 1]

        pos_audio, sr = torchaudio.load(pos_path)
        neg_audio, sr = torchaudio.load(neg_path)

        pos_outputs = encoder(pos_audio.to(encoder.device))
        neg_outputs = encoder(neg_audio.to(encoder.device))

        with open(Path(pos_path).with_suffix(".txt")) as f:
            pos_text = f.read().strip()

        with open(Path(neg_path).with_suffix(".txt")) as f:
            neg_text = f.read().strip()

        example = {
            "filename": {
                "pos": str(Path(pos_path).relative_to(SC_dir).with_suffix("")),
                "neg": str(Path(neg_path).relative_to(SC_dir).with_suffix("")),
            },
            "units": {
                "pos": pos_outputs[0]["units"].tolist(),
                "neg": neg_outputs[0]["units"].tolist(),
            },
            "text": {
                "pos": pos_text,
                "neg": neg_text,
            },
        }
        SC_test.append(example)

    return Dataset.from_list(SC_test)


def tokenize_eval(config):
    tqdm.pandas()

    app_dir = Path(config.dataset.APP_DIR).expanduser()
    tSC_dir = Path(config.dataset.tSC_DIR)
    sSC_dir = Path(config.dataset.sSC_DIR)

    swuggy_dev_dir = app_dir / "datasets/sLM21-dataset/lexical/dev"
    sblimp_dev_dir = app_dir / "datasets/sLM21-dataset/syntactic/dev"
    swuggy_test_dir = app_dir / "datasets/sLM21-dataset/lexical/test"
    sblimp_test_dir = app_dir / "datasets/sLM21-dataset/syntactic/test"

    encoder = SylRegForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")

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

    tSC_test = tokenize_storycloze(encoder, tSC_dir)
    sSC_test = tokenize_storycloze(encoder, sSC_dir)

    swuggy = DatasetDict({"validation": swuggy_dev, "test": swuggy_test})
    sblimp = DatasetDict({"validation": sblimp_dev, "test": sblimp_test})
    tSC = DatasetDict({"test": tSC_test})
    sSC = DatasetDict({"test": sSC_test})

    swuggy.push_to_hub(config.dataset.name, "sWUGGY")
    sblimp.push_to_hub(config.dataset.name, "sBLIMP")
    tSC.push_to_hub(config.dataset.name, "tSC")
    sSC.push_to_hub(config.dataset.name, "sSC")


class ForcedAligner:
    def __init__(self, aligner_name: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf"):
        self.processor = AutoProcessor.from_pretrained(aligner_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            aligner_name, dtype=torch.bfloat16, device_map="auto"
        )

    def clean_token_punctuation(self, token: str) -> str:
        word = "".join(ch for ch in token if _is_kept_char(ch))

        if len(word) < 2:
            return word

        if _is_kept_char(token[-2]) and token[-1] in {",", ".", "?", "!"}:
            return word + token[-1]

        return word

    def split_segment_with_chinese(self, seg: str) -> list[str]:
        tokens: list[str] = []
        buf: list[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if _is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush_buf()
        return tokens

    def tokenize_space_lang_punctuation(self, text: str) -> list[str]:
        tokens: list[str] = []
        for seg in text.split():
            cleaned = self.clean_token_punctuation(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    @torch.inference_mode()
    def __call__(self, input_values: torch.Tensor, text: str) -> list[dict[str, Any]]:
        # Step 1: Prepare alignment inputs
        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=input_values.squeeze(0).numpy(),
            transcript=text,
            language="English",
        )
        inputs = inputs.to(self.model.device, self.model.dtype)

        # Step 2: Run forced aligner
        aligner_outputs = self.model(**inputs)

        # Step 3: Decode timestamps
        timestamps = self.processor.decode_forced_alignment(
            logits=aligner_outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id,
        )[0]

        word_lists_punctuation = self.tokenize_space_lang_punctuation(text)
        assert len(word_lists) == len(word_lists_punctuation)

        aligned_text = [
            {
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "word": " " + word,  # prepend a space for concatenation. See Line 300.
            }
            for item, word in zip(timestamps, word_lists_punctuation, strict=True)
        ]
        return aligned_text


def add_aligned_units(example: dict[str, Any]) -> dict[str, Any]:
    if not example["aligned_text"]:
        example["aligned_units"] = []
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


def align_units(config, num_shards: int = 1):
    data_files = [
        f"{config.dataset.manifest_prefix}_with_alignment{shard_index}.json" for shard_index in range(num_shards)
    ]
    dataset = load_dataset("json", data_files=data_files, split="train")
    dataset = DatasetDict({"train": dataset})
    dataset.push_to_hub(config.dataset.name, "Libri-Light")
