import random

import torch
import torchaudio
from datasets import Audio, concatenate_datasets, load_dataset
from kokoro import KPipeline
from omegaconf import OmegaConf
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from ..s5hubert import SylRegForSyllableDiscovery
from .data.tinystories import oov_pattern


def get_synthesizer(model_name_or_path):
    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")
    pipeline = KPipeline(lang_code="a")

    @torch.inference_mode()
    def synthesize(example):
        messages = []
        audio = []

        user_voice = random.choice(["af_bella", "af_nicole", "am_michael", "am_puck"])

        for message in example["messages"]:
            voice = "af_heart" if message["role"] == "assistant" else user_voice
            generator = pipeline(message["content"], voice=voice)

            input_values = torch.cat([input_values for _, (gs, _, input_values) in enumerate(generator)])
            audio.append(input_values)
            input_values = torchaudio.functional.resample(input_values, 24000, 16000).unsqueeze(0)

            content = encoder(input_values.to(encoder.device))[0]["units"]
            message["content"] = "".join(f"<{unit}>" for unit in content)
            messages.append(message)

        example["spoken_messages"] = messages
        example["audio"] = {
            "array": torchaudio.functional.resample(torch.cat(audio), 24000, 16000).numpy(),
            "sampling_rate": 16000,
        }
        return example

    return synthesize


def data(config, num_proc: int = 6):
    dataset = load_dataset(
        "HuggingFaceTB/smoltalk2", "SFT", split="smoltalk_smollm3_everyday_conversations_no_think", num_proc=num_proc
    )
    dataset = dataset.filter(
        lambda example: not oov_pattern.search("".join(message["content"] for message in example["messages"])),
        num_proc=num_proc,
    )
    dataset = dataset.map(get_synthesizer(config.speech2unit.model_name_or_path))
    dataset = dataset.cast_column("audio", Audio())
    dataset.push_to_hub(config.dataset.name, split="train")


def finetune(config):
    args = SFTConfig(**OmegaConf.to_container(config.training_args))

    train_dataset = load_dataset(config.dataset.name, split="train")
    train_dataset = concatenate_datasets(
        [
            train_dataset,
            train_dataset.remove_columns("messages").rename_column("spoken_messages", "messages"),
        ]
    )

    trainer = SFTTrainer(
        model=config.model_args.name,
        args=args,
        train_dataset=train_dataset,
        peft_config=LoraConfig(),
    )
    trainer.train()
