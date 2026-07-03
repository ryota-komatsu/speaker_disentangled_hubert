import json
import re
from pathlib import Path

import torchaudio
from datasets import load_dataset
from kokoro import KPipeline
from tqdm import tqdm

from ...s5hubert import SylRegForSyllableDiscovery
from .utils import add_aligned_units, get_aligner

vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\",.?! ;:0123456789-%"
oov_pattern = re.compile(f"[^{re.escape(vocab)}]")


def tokenize_tinystories(
    num_shards: int = 1,
    shard_index: int = 0,
    data_dir: str = "data/tinystories",
    model_name_or_path: str = "ryota-komatsu/SylReg-Distill",
    num_proc: int = 6,
):
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.filter(lambda example: not oov_pattern.search(example["text"]), num_proc=num_proc)
    dataset = dataset.shard(num_shards, shard_index)

    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")
    aligner = get_aligner()

    pipeline = KPipeline(lang_code="a")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / f"manifest{shard_index}.json"

    with open(manifest_path, "w") as f:
        for i, example in enumerate(tqdm(dataset)):
            text = re.sub(r"\s+", " ", example["text"])

            generator = pipeline(text, voice="af_heart")

            for j, (gs, _, input_values) in enumerate(generator):
                id_ = f"audio{shard_index}/audio{shard_index}_{i:07}_{j:03}"

                input_values = torchaudio.functional.resample(input_values, 24000, 16000).unsqueeze(0)

                outputs = encoder(input_values.to(encoder.device))

                example = {
                    "id": id_,
                    "units": outputs[0]["units"].tolist(),
                    "durations": outputs[0]["durations"].tolist(),
                    "aligned_text": aligner(input_values, gs),
                }
                example = add_aligned_units(example)
                json.dump(example, f)
                f.write("\n")
