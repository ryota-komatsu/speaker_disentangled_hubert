import json
import math
import os
import re
from pathlib import Path

import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import ADDITIONAL_DIACRITICS

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


def tokenize_librilight(config, num_shards: int = 1, shard_index: int = 0):
    from ...s5hubert import S5HubertForSyllableDiscovery

    data_files = [
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_small.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_medium.jsonl.gz"),
        os.path.join(config.dataset.lh_dir, "libriheavy_cuts_large.jsonl.gz"),
    ]
    dataset = load_dataset("json", data_files=data_files, split="train")
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

            example = {
                "audio_filepath": save_path,
                "text": text,
                "id": example["id"],
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
                "intermediate_units": outputs[0]["intermediate_units"].tolist(),
            }
            json.dump(example, f)
            f.write("\n")
