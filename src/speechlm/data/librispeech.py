from datasets import concatenate_datasets, load_dataset

from ...s5hubert import SylRegForSyllableDiscovery


def get_map_fn(model_name_or_path: str = "ryota-komatsu/SylReg-Distill"):
    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    def map_fn(example):
        input_values = example["audio"]["array"].unsqueeze(0)

        outputs = encoder(input_values.to(encoder.device))

        example = {
            "units": outputs[0]["units"].tolist(),
            "durations": outputs[0]["durations"].tolist(),
        }
        return example

    return map_fn


def tokenize_librispeech(model_name_or_path: str = "ryota-komatsu/SylReg-Distill", num_proc: int = 6):
    dataset = concatenate_datasets(
        [
            load_dataset("openslr/librispeech_asr", "all", split="train.clean.100", num_proc=num_proc),
            load_dataset("openslr/librispeech_asr", "all", split="train.clean.360", num_proc=num_proc),
            load_dataset("openslr/librispeech_asr", "all", split="train.other.500", num_proc=num_proc),
        ]
    )
    dataset = dataset.with_format("torch")
    dataset = dataset.map(
        get_map_fn(model_name_or_path),
        remove_columns=["file", "audio", "text", "speaker_id", "chapter_id"],
    )
    dataset.push_to_hub("ryota-komatsu/SylReg", "LibriSpeech")
