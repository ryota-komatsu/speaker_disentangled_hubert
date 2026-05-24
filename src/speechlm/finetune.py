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


def get_synthesizer(model_name_or_path: str):
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


def get_dailytalk_tokenizer(model_name_or_path: str):
    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    def tokenize(example):
        messages = []
        spoken_messages = []

        for turn_idx, (text, audio_cut_idx) in enumerate(zip(example["texts"], example["audio_cut_idxs"], strict=True)):
            input_values = example["audio"]["array"][audio_cut_idx[0] : audio_cut_idx[1]]
            input_values = torchaudio.functional.resample(
                input_values, example["audio"]["sampling_rate"], 16000
            ).unsqueeze(0)

            role = "user" if turn_idx % 2 == 0 else "assistant"
            content = encoder(input_values.to(encoder.device))[0]["units"]

            message = {"role": role, "content": "".join(f"<{unit}>" for unit in content)}
            spoken_message = {"role": role, "content": text}

            messages.append(message)
            spoken_messages.append(spoken_message)

        return {"messages": messages, "spoken_messages": spoken_messages}

    return tokenize


def data(config, num_proc: int = 6):
    dataset = load_dataset(
        "HuggingFaceTB/smoltalk2", "SFT", split="smoltalk_smollm3_everyday_conversations_no_think", num_proc=num_proc
    )
    dataset = dataset.filter(
        lambda example: not oov_pattern.search("".join(message["content"] for message in example["messages"])),
        num_proc=num_proc,
    )
    dataset = dataset.map(
        get_synthesizer(config.speech2unit.model_name_or_path), remove_columns=["chat_template_kwargs"]
    )
    dataset = dataset.cast_column("audio", Audio())
    dataset.push_to_hub(config.dataset.name, "smoltalk2", split="train")

    dataset = load_dataset("eustlb/dailytalk-conversations-grouped", split="train", num_proc=num_proc)
    dataset = dataset.with_format("torch")
    dataset = dataset.map(
        get_dailytalk_tokenizer(config.speech2unit.model_name_or_path),
        remove_columns=["conversation_id", "speaker_ids", "turn_ids", "texts", "audio_cut_idxs", "conversation"],
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.push_to_hub(config.dataset.name, "dailytalk", split="train")


def finetune(config):
    args = SFTConfig(**OmegaConf.to_container(config.training_args))

    smoltalk2 = load_dataset(config.dataset.name, "smoltalk2", split="train").remove_columns("audio")
    dailytalk = load_dataset(config.dataset.name, "dailytalk", split="train").remove_columns("audio")

    train_dataset = concatenate_datasets(
        [
            smoltalk2,
            dailytalk,
            smoltalk2.remove_columns("messages").rename_column("spoken_messages", "messages"),
            dailytalk.remove_columns("messages").rename_column("spoken_messages", "messages"),
        ]
    )

    trainer = SFTTrainer(
        model=config.model_args.name,
        args=args,
        train_dataset=train_dataset,
        peft_config=LoraConfig(),
    )
    trainer.train()
