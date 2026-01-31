import json
from pathlib import Path

import torchaudio
from datasets import load_dataset
from tqdm import tqdm

from ...s5hubert import S5HubertForSyllableDiscovery


def tokenize_voxpopuli(
    data_dir: str = "data/voxpopuli",
    model_name_or_path: str = "ryota-komatsu/s5-hubert",
):
    dataset = load_dataset("facebook/voxpopuli", "en", split="train", trust_remote_code=True)
    dataset = dataset.with_format("torch")

    encoder = S5HubertForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / "manifest.json"

    with open(manifest_path, "w") as f:
        for example in tqdm(dataset):
            audio_filepath = (Path(data_dir) / example["audio_id"]).with_suffix(".flac")
            audio_filepath.parent.mkdir(parents=True, exist_ok=True)
            audio_filepath = str(audio_filepath)

            input_values = torchaudio.functional.resample(
                example["audio"]["array"], example["audio"]["sampling_rate"], 16000
            ).unsqueeze(0)
            torchaudio.save(audio_filepath, input_values, 16000, encoding="PCM_S", bits_per_sample=16)

            outputs = encoder(input_values.to(encoder.device))

            example = {
                # "audio_filepath": audio_filepath,
                # "text": example["normalized_text"],
                "id": example["audio_id"],
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
            }
            json.dump(example, f)
            f.write("\n")
