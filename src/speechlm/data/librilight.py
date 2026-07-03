# from https://github.com/facebookresearch/libri-light/blob/main/data_preparation/cut_by_vad.py

# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
import json
import math
import os
import re
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import ADDITIONAL_DIACRITICS

from ...s5hubert import SylRegForSyllableDiscovery
from .utils import ForcedAligner, add_aligned_units

vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\",.?! ;:()[]—_" + "".join(ADDITIONAL_DIACRITICS)
pattern = f"[^{re.escape(vocab)}]"


def normalize_text(s: str) -> str:
    s = s.replace("‘", "'")
    s = s.replace("’", "'")
    tokens = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',.?")
    s_list = [x if x in tokens else ADDITIONAL_DIACRITICS.get(x, " ") for x in s]
    s = " ".join("".join(s_list).split()).strip()

    s = re.sub(r"\baround'em\b", "around them", s)
    s = re.sub(r"\bb'lieve\b", "believe", s)
    s = re.sub(r"\bbewilder'd\b", "bewildered", s)
    s = re.sub(r"\bcap'n\b", "captain", s)
    s = re.sub(r"\bCap'n\b", "Captain", s)
    s = re.sub(r"\bcharm'em\b", "charm them", s)
    s = re.sub(r"\bdiff'rence\b", "difference", s)
    s = re.sub(r"\be'en\b", "even", s)
    s = re.sub(r"\bfetchin'\s", "fetching ", s)
    s = re.sub(r"\bgive'em\b", "give them", s)
    s = re.sub(r"\binv'tation", "invitation", s)
    s = re.sub(r"\bmore'n\b", "more than", s)
    s = re.sub(r"\bof'em\b", "of them", s)
    s = re.sub(r"\bop'ning\b", "opening", s)
    s = re.sub(r"\bpass'd\b", "passed", s)
    s = re.sub(r"\bp'raps\b", "perhaps", s)
    s = re.sub(r"\bshorten'd\b", "shortened", s)
    s = re.sub(r"\bs'pose\b", "suppose", s)
    s = re.sub(r"\btellin'\s", "telling ", s)
    s = re.sub(r"\bvisitin'\s", "visiting ", s)
    s = re.sub(r"\bwith'em\b", "with them", s)

    s = re.sub(r"\s\?", "?", s)
    s = re.sub(r"\s,", ",", s)

    return s


def tokenize_librilight(
    num_shards: int = 1,
    shard_index: int = 0,
    data_dir: str = "data/librilight",
    model_name_or_path: str = "ryota-komatsu/SylReg-Distill",
    tgt_len_sec: int = 25,
    min_len_sec: int = 5,
    max_len_sec: int = 30,
):
    tgt_chunk_size = tgt_len_sec * 16000 + 80
    min_chunk_size = min_len_sec * 16000 + 80
    max_chunk_size = max_len_sec * 16000 + 80

    data_files = list(glob.glob(os.path.join(data_dir, "*/*/*/*.json")))
    shard_size = (len(data_files) // num_shards) + 1
    dataset = data_files[shard_index * shard_size : (shard_index + 1) * shard_size]

    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    manifest_path = Path(data_dir) / f"manifest{shard_index}.json"

    with open(manifest_path, "w") as f:
        for data_file in tqdm(dataset):
            with open(data_file) as g:
                example = json.load(g)

            audio_filepath = Path(data_file).with_suffix(".flac")
            data, sr = torchaudio.load(audio_filepath)

            chunks = []
            to_stitch = []
            length_accumulated = 0.0

            # cut by VAD
            for start, end in example["voice_activity"]:
                start_index = int(start * 16000)
                end_index = int(end * 16000)
                slice = data[:, start_index:end_index]

                if length_accumulated + (end - start) > tgt_len_sec and length_accumulated > 0:
                    input_values = torch.cat(to_stitch, dim=1)

                    if input_values.shape[1] < max_chunk_size:
                        chunks.append(input_values)
                    else:
                        input_values = list(torch.split(input_values, tgt_chunk_size, dim=1))

                        if len(input_values) > 1 and input_values[-1].shape[1] < min_chunk_size:
                            input_values[-2] = torch.cat([input_values[-2], input_values[-1]], dim=1)
                            input_values.pop()

                        chunks.extend(input_values)

                    to_stitch = []
                    length_accumulated = 0.0

                to_stitch.append(slice)
                length_accumulated += end - start

            # last chunk
            if to_stitch:
                input_values = torch.cat(to_stitch, dim=1)

                if input_values.shape[1] < max_chunk_size:
                    chunks.append(input_values)
                else:
                    input_values = list(torch.split(input_values, tgt_chunk_size, dim=1))

                    if len(input_values) > 1 and input_values[-1].shape[1] < min_chunk_size:
                        input_values[-2] = torch.cat([input_values[-2], input_values[-1]], dim=1)
                        input_values.pop()

                    chunks.extend(input_values)

            # tokenize
            for chunk_index, input_values in enumerate(chunks):
                id_ = str(Path(data_file).relative_to(data_dir).with_suffix("")) + f"_{chunk_index}"
                outputs = encoder(input_values.to(encoder.device))

                example = {
                    "id": id_,
                    "units": outputs[0]["units"].tolist(),
                    "durations": outputs[0]["durations"].tolist(),
                }
                json.dump(example, f)
                f.write("\n")


def tokenize_libriheavy(
    config,
    num_shards: int = 1,
    shard_index: int = 0,
    data_dir: str = "data/libriheavy",
):
    data_files = [
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_small.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_medium.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_large.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_dev.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_test_clean.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_test_other.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_test_clean_large.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_test_other_large.jsonl.gz"),
    ]
    dataset = load_dataset("json", data_files=data_files, split="train")
    dataset = dataset.shard(num_shards=num_shards, index=shard_index)

    encoder = SylRegForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")
    aligner = ForcedAligner()

    manifest_path = Path(data_dir) / f"manifest{shard_index}.json"

    with open(manifest_path, "w") as f:
        for example in tqdm(dataset):
            load_path = os.path.join(config.dataset.ll_dir, example["recording"]["id"] + config.dataset.ext_audio)
            save_path = os.path.join(config.dataset.lh_dir, example["id"] + config.dataset.ext_audio)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            input_values, sr = torchaudio.load(
                load_path,
                frame_offset=math.floor(16000 * max(example["start"], 0)),
                num_frames=math.floor(16000 * example["duration"]),
            )
            torchaudio.save(save_path, input_values, 16000, bits_per_sample=16)

            outputs = encoder(input_values.to(encoder.device))

            text = example["supervisions"][0]["custom"]["texts"][0]
            text = normalize_text(text)

            example = {
                "id": example["id"],
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
                "aligned_text": aligner(input_values, text),
            }
            example = add_aligned_units(example)
            json.dump(example, f)
            f.write("\n")
