import json
import re
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from kokoro import KPipeline
from tqdm import tqdm

from ...s5hubert import SylRegForSyllableDiscovery

vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\",.?! ;:0123456789-%"
oov_pattern = re.compile(f"[^{re.escape(vocab)}]")


def tokenize_tinystories(
    num_shards: int = 1,
    shard_index: int = 0,
    data_dir: str = "data/tinystories",
    model_name_or_path: str = "ryota-komatsu/SylReg-Distill",
):
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.shard(num_shards, shard_index)

    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    pipeline = KPipeline(lang_code="a")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / f"manifest{shard_index}.json"

    with open(manifest_path, "w") as f:
        for i, example in enumerate(tqdm(dataset)):
            text = re.sub(r"\s+", " ", example["text"])

            if oov_pattern.search(text):
                continue

            generator = pipeline(text, voice="af_heart")

            for j, (gs, _, input_values) in enumerate(generator):
                id_ = f"audio{shard_index}/audio{shard_index}_{i:07}_{j:03}"
                audio_filepath = (Path(data_dir) / id_).with_suffix(".flac")
                audio_filepath.parent.mkdir(parents=True, exist_ok=True)
                audio_filepath = str(audio_filepath)

                input_values = torchaudio.functional.resample(input_values, 24000, 16000).unsqueeze(0)

                outputs = encoder(input_values.to(encoder.device))

                example = {
                    # "text": gs,
                    "id": id_,
                    "units": outputs[0]["units"].tolist(),
                    "durations": outputs[0]["durations"].tolist(),
                }
                json.dump(example, f)
                f.write("\n")
