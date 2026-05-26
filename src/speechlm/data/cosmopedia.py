import json
import re
from pathlib import Path

import torchaudio
from datasets import load_dataset
from kokoro import KPipeline
from tqdm import tqdm

from ...s5hubert import SylRegForSyllableDiscovery

vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\",.?! ;-"
oov_pattern = re.compile(f"[^{re.escape(vocab)}]")
enum_pattern = re.compile(r"\b[IVX]+[.)]")
line_pattern = re.compile(r"[\-]{2,}")


def filter_fn(example: dict) -> bool:
    return (
        not oov_pattern.search(example["text"])
        and not enum_pattern.search(example["text"])
        and not line_pattern.search(example["text"])
        and example["seed_data"] != "auto_math_text"
        and example["format"] not in {"dialogue", "story"}
    )


def tokenize_cosmopedia(
    num_shards: int = 1,
    shard_index: int = 0,
    data_dir: str = "data/cosmopedia",
    model_name_or_path: str = "ryota-komatsu/SylReg-Distill",
    num_proc: int = 6,
):
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", num_proc=num_proc)
    dataset = dataset.filter(filter_fn, num_proc=num_proc)
    dataset = dataset.shard(num_shards, shard_index)

    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    pipeline = KPipeline(lang_code="a")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / f"manifest{shard_index}.json"

    with open(manifest_path, "w") as f:
        for i, example in enumerate(tqdm(dataset)):
            text = re.sub(r"\s+", " ", example["text"])

            generator = pipeline(text, voice="af_heart")

            for j, (gs, _, input_values) in enumerate(generator):
                id_ = f"audio{shard_index}/audio{shard_index}_{i:08}_{j:08}"

                input_values = torchaudio.functional.resample(input_values, 24000, 16000).unsqueeze(0)

                outputs = encoder(input_values.to(encoder.device))

                example = {
                    "id": id_,
                    "units": outputs[0]["units"].tolist(),
                    "durations": outputs[0]["durations"].tolist(),
                }
                json.dump(example, f)
                f.write("\n")
