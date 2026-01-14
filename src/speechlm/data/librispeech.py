import json
from pathlib import Path

import torchaudio
from tqdm import tqdm

from ...s5hubert import S5HubertForSyllableDiscovery


def tokenize_librispeech(
    data_dir: str = "data/LibriSpeech",
    model_name_or_path: str = "ryota-komatsu/s5-hubert",
):
    dataset = Path(data_dir).glob("train-*/**/*.flac")

    encoder = S5HubertForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    manifest_path = Path(data_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        for audio_filepath in tqdm(dataset):
            id_ = str(audio_filepath.relative_to(data_dir).with_suffix(""))
            split, speaker_id, chap_id, utterance_id = id_.split("/")
            file = Path(data_dir) / split / speaker_id / chap_id / f"{speaker_id}-{chap_id}.trans.txt"

            with open(file) as f:
                for line in f:
                    text_id, text = line.rstrip().split(" ", maxsplit=1)
                    if text_id == utterance_id:
                        break

            audio_filepath = str(audio_filepath)

            input_values, sr = torchaudio.load(audio_filepath)

            outputs = encoder(input_values.to(encoder.device))

            example = {
                "audio_filepath": audio_filepath,
                "text": text.lower(),
                "id": id_,
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
                "intermediate_units": outputs[0]["intermediate_units"].tolist(),
            }
            json.dump(example, f)
            f.write("\n")
