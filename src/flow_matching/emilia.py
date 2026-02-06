# from https://github.com/SWivid/F5-TTS/blob/1.1.15/src/f5_tts/train/datasets/prepare_emilia.py

# MIT License
#
# Copyright (c) 2024 Yushen CHEN
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

import json
import re
from pathlib import Path

import torchaudio
from datasets import Array2D, Features, List, Value, load_dataset
from tqdm import tqdm

from ..bigvgan.data import mel_spectrogram
from ..s5hubert import S5HubertForSyllableDiscovery

out_emilia = {
    "EN_B00013_S00913",
    "EN_B00042_S00120",
    "EN_B00055_S04111",
    "EN_B00061_S00693",
    "EN_B00061_S01494",
    "EN_B00061_S03375",
    "EN_B00059_S00092",
    "EN_B00111_S04300",
    "EN_B00100_S03759",
    "EN_B00087_S03811",
    "EN_B00059_S00950",
    "EN_B00089_S00946",
    "EN_B00078_S05127",
    "EN_B00070_S04089",
    "EN_B00074_S09659",
    "EN_B00061_S06983",
    "EN_B00061_S07060",
    "EN_B00059_S08397",
    "EN_B00082_S06192",
    "EN_B00091_S01238",
    "EN_B00089_S07349",
    "EN_B00070_S04343",
    "EN_B00061_S02400",
    "EN_B00076_S01262",
    "EN_B00068_S06467",
    "EN_B00076_S02943",
    "EN_B00064_S05954",
    "EN_B00061_S05386",
    "EN_B00066_S06544",
    "EN_B00076_S06944",
    "EN_B00072_S08620",
    "EN_B00076_S07135",
    "EN_B00076_S09127",
    "EN_B00065_S00497",
    "EN_B00059_S06227",
    "EN_B00063_S02859",
    "EN_B00075_S01547",
    "EN_B00061_S08286",
    "EN_B00079_S02901",
    "EN_B00092_S03643",
    "EN_B00096_S08653",
    "EN_B00063_S04297",
    "EN_B00063_S04614",
    "EN_B00079_S04698",
    "EN_B00104_S01666",
    "EN_B00061_S09504",
    "EN_B00061_S09694",
    "EN_B00065_S05444",
    "EN_B00063_S06860",
    "EN_B00065_S05725",
    "EN_B00069_S07628",
    "EN_B00083_S03875",
    "EN_B00071_S07665",
    "EN_B00071_S07665",
    "EN_B00062_S04187",
    "EN_B00065_S09873",
    "EN_B00065_S09922",
    "EN_B00084_S02463",
    "EN_B00067_S05066",
    "EN_B00106_S08060",
    "EN_B00073_S06399",
    "EN_B00073_S09236",
    "EN_B00087_S00432",
    "EN_B00085_S05618",
    "EN_B00064_S01262",
    "EN_B00072_S01739",
    "EN_B00059_S03913",
    "EN_B00069_S04036",
    "EN_B00067_S05623",
    "EN_B00060_S05389",
    "EN_B00060_S07290",
    "EN_B00062_S08995",
    "EN_B00079_S00298",
    "EN_B00000_S00220",
    "EN_B00000_S00250",
}

out_yodas = {
    "EN_tcGXd8kka3c_SPEAKER_00",
    "EN_tcytC3CYuJ0_SPEAKER_01",
    "EN_tcytC3CYuJ0_SPEAKER_05",
}

vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\",.?! ;:0123456789-%"
oov_pattern = re.compile(f"[^{re.escape(vocab)}]")


def tokenize_emilia(
    num_shards: int = 1,
    shard_index: int = 0,
    data_dir: str = "data/emilia",
    model_name_or_path: str = "ryota-komatsu/s5-hubert",
):
    dataset = load_dataset("amphion/Emilia-Dataset", data_files={"en": "Emilia/EN/*.tar"}, split="en", streaming=True)
    dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.with_format("torch")

    encoder = S5HubertForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / f"manifest{shard_index}.json"

    with open(manifest_path, "w") as f:
        for example in tqdm(dataset):
            text = re.sub(r"\s+", " ", example["json"]["text"])

            if (
                oov_pattern.search(text)
                or example["json"]["dnsmos"] < 3.45
                or example["json"]["duration"] < 10
                or example["json"]["wav"].split("/")[1] in out_emilia
            ):
                continue

            id_ = str(Path(example["json"]["wav"]).with_suffix(""))
            audio_filepath = (Path(data_dir) / id_).with_suffix(".flac")
            audio_filepath.parent.mkdir(parents=True, exist_ok=True)
            audio_filepath = str(audio_filepath)

            input_values = torchaudio.functional.resample(
                example["mp3"]["array"], example["mp3"]["sampling_rate"], 16000
            ).unsqueeze(0)
            torchaudio.save(audio_filepath, input_values, 16000, encoding="PCM_S", bits_per_sample=16)

            outputs = encoder(input_values.to(encoder.device))

            example = {
                "audio_filepath": audio_filepath,
                "text": text,
                "id": id_,
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
            }
            json.dump(example, f)
            f.write("\n")


def tokenize_yodas(
    num_shards: int = 1,
    shard_index: int = 0,
    data_dir: str = "data/emilia_yodas",
    model_name_or_path: str = "ryota-komatsu/s5-hubert",
):
    dataset = load_dataset(
        "amphion/Emilia-Dataset", data_files={"en": "Emilia-YODAS/EN/*.tar"}, split="en", streaming=True
    )
    dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.with_format("torch")

    encoder = S5HubertForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_dir) / f"manifest{shard_index}.json"

    with open(manifest_path, "w") as f:
        for example in tqdm(dataset):
            text = re.sub(r"\s+", " ", example["json"]["text"])

            if (
                oov_pattern.search(text)
                or example["json"]["dnsmos"] < 3.45
                or example["json"]["duration"] < 10
                or example["json"]["speaker"] in out_yodas
            ):
                continue

            id_ = str(Path(example["__url__"]).stem / example["json"]["speaker"] / example["json"]["_id"])
            audio_filepath = (Path(data_dir) / id_).with_suffix(".flac")
            audio_filepath.parent.mkdir(parents=True, exist_ok=True)
            audio_filepath = str(audio_filepath)

            input_values = torchaudio.functional.resample(
                example["mp3"]["array"], example["mp3"]["sampling_rate"], 16000
            ).unsqueeze(0)
            torchaudio.save(audio_filepath, input_values, 16000, encoding="PCM_S", bits_per_sample=16)

            outputs = encoder(input_values.to(encoder.device))

            example = {
                "audio_filepath": audio_filepath,
                "text": text,
                "id": id_,
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
            }
            json.dump(example, f)
            f.write("\n")


def get_tokenize_fn(data_dir):
    data_dir = Path(data_dir).resolve()

    def _add_spectrogram(example):
        audio_filepath = str((data_dir / example["id"]).with_suffix(".flac"))
        input_values, sr = torchaudio.load(audio_filepath)
        input_values = input_values.cuda()
        input_values = input_values / input_values.abs().max() * 0.95
        input_values = input_values.unsqueeze(0)

        spectrogram_labels = mel_spectrogram(input_values).squeeze(0)  # (80, len)
        spectrogram_labels = spectrogram_labels.transpose(0, 1)  # (len, 80)
        spectrogram_labels = spectrogram_labels.cpu().tolist()

        return {"spectrogram": spectrogram_labels}

    return _add_spectrogram


def add_spectrogram(
    dataset_name: str,
    dataset_path: str,
    num_proc: int = 16,
    config_name: str = "emilia",
    data_dir: str = "data/emilia",
):
    features = Features(
        {
            "id": Value("string"),
            "units": List(Value("int32")),
            "durations": List(Value("int32")),
            "spectrogram": Array2D(shape=(None, 80), dtype="float32"),
        }
    )

    dataset = load_dataset(dataset_name, config_name, split="train")
    dataset = dataset.map(
        get_tokenize_fn(data_dir),
        num_proc=num_proc,
        remove_columns=["intermediate_units"],
    )
    dataset = dataset.cast(features, num_proc=num_proc)
    dataset.save_to_disk(dataset_path)
