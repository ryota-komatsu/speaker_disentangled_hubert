from datasets import Audio, load_dataset

from ...s5hubert import SylRegForSyllableDiscovery
from .utils import ForcedAligner, add_aligned_units


def get_map_fn(model_name_or_path: str = "ryota-komatsu/SylReg-Distill"):
    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")
    aligner = ForcedAligner()

    def map_fn(example):
        input_values = example["audio"]["array"].unsqueeze(0)

        outputs = encoder(input_values.to(encoder.device))

        example = {
            # "text": example["normalized_text"],
            "id": example["audio_id"],
            "units": outputs[0]["units"].tolist(),
            "durations": outputs[0]["durations"].tolist(),
            "aligned_text": aligner(input_values, example["normalized_text"]),
        }
        example = add_aligned_units(example)
        return example

    return map_fn


def tokenize_voxpopuli(model_name_or_path: str = "ryota-komatsu/SylReg-Distill", num_proc: int = 6):
    dataset = load_dataset("facebook/voxpopuli", "en", split="train", trust_remote_code=True, num_proc=num_proc)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.with_format("torch")
    dataset = dataset.map(
        get_map_fn(model_name_or_path),
        remove_columns=[
            "audio_id",
            "language",
            "audio",
            "raw_text",
            "normalized_text",
            "gender",
            "speaker_id",
            "is_gold_transcript",
            "accent",
        ],
    )
    dataset.push_to_hub("ryota-komatsu/SylReg", "voxpopuli")
