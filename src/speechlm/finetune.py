import random

import librosa
import torch
import torchaudio
from datasets import Audio, concatenate_datasets, load_dataset
from kokoro import KPipeline
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
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

        user_voice = random.choice(["af_bella", "af_aoede", "am_michael", "am_puck"])

        for message in example["messages"]:
            voice = "af_heart" if message["role"] == "assistant" else user_voice
            generator = pipeline(message["content"], voice=voice)

            input_values = torch.cat([input_values for _, (gs, _, input_values) in enumerate(generator)])
            input_values = librosa.effects.trim(input_values.numpy(), top_db=20)[0]
            input_values = torch.from_numpy(input_values)
            input_values = torchaudio.functional.resample(input_values, 24000, 16000)
            audio.append(input_values)

            content = encoder(input_values.unsqueeze(0).to(encoder.device))[0]["units"]
            messages.append({"role": message["role"], "content": "".join(f"<{unit}>" for unit in content)})

        example["spoken_messages"] = messages
        example["audio"] = {"array": torch.cat(audio).numpy(), "sampling_rate": 16000}
        return example

    return synthesize


def get_dailytalk_tokenizer(model_name_or_path: str):
    encoder = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path, device_map="cuda")

    def tokenize(example):
        messages = []
        spoken_messages = []

        for turn_idx, (text, audio_cut_idx) in enumerate(zip(example["texts"], example["audio_cut_idxs"], strict=True)):
            input_values = example["audio"]["array"][audio_cut_idx[0] : audio_cut_idx[1]]
            input_values = torchaudio.functional.resample(input_values, example["audio"]["sampling_rate"], 16000)

            role = "user" if turn_idx % 2 == 0 else "assistant"
            content = encoder(input_values.unsqueeze(0).to(encoder.device))[0]["units"]

            message = {"role": role, "content": "".join(f"<{unit}>" for unit in content)}
            spoken_message = {"role": role, "content": text}

            messages.append(message)
            spoken_messages.append(spoken_message)

        return {"messages": messages, "spoken_messages": spoken_messages}

    return tokenize


def filter_fn(example):
    return (
        not oov_pattern.search("".join(message["content"] for message in example["messages"]))
        and len(example["messages"][0]["content"]) < 128
        and len(example["messages"][1]["content"]) < 32
    )


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
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.push_to_hub(config.dataset.name, "everyday-conversations", split="train")

    # OpenHermes
    dataset = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split="OpenHermes_2.5_no_think", num_proc=num_proc)
    dataset = dataset.filter(filter_fn, num_proc=num_proc)
    dataset = dataset.map(
        get_synthesizer(config.speech2unit.model_name_or_path),
        remove_columns=["chat_template_kwargs", "source"],
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.push_to_hub(config.dataset.name, "OpenHermes", split="train")

    # dailytalk
    dataset = load_dataset("eustlb/dailytalk-conversations-grouped", split="train", num_proc=num_proc)
    dataset = dataset.with_format("torch")
    dataset = dataset.map(
        get_dailytalk_tokenizer(config.speech2unit.model_name_or_path),
        remove_columns=["conversation_id", "speaker_ids", "turn_ids", "texts", "audio_cut_idxs", "conversation"],
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.push_to_hub(config.dataset.name, "dailytalk", split="train")


def interleave(example):
    speech_to_text = []
    text_to_speech = []

    for turn_idx, (message, spoken_message) in enumerate(zip(example["messages"], example["spoken_messages"])):
        if turn_idx % 2 == 0:
            speech_to_text.append(spoken_message)
            text_to_speech.append(message)
        else:
            speech_to_text.append(message)
            text_to_speech.append(spoken_message)

    example["speech_to_text_messages"] = speech_to_text
    example["text_to_speech_messages"] = text_to_speech

    return example


def finetune(config):
    args = SFTConfig(**OmegaConf.to_container(config.training_args))

    everyday_conversations = load_dataset(config.dataset.name, "everyday-conversations", split="train").remove_columns(
        "audio"
    )
    dailytalk = load_dataset(config.dataset.name, "dailytalk", split="train").remove_columns("audio")
    openhermes = load_dataset(config.dataset.name, "OpenHermes", split="train").remove_columns("audio")

    everyday_conversations = everyday_conversations.map(interleave)
    dailytalk = dailytalk.map(interleave)
    openhermes = openhermes.map(interleave)

    train_dataset = concatenate_datasets(
        [
            everyday_conversations,
            dailytalk,
            openhermes,
            everyday_conversations.remove_columns("messages").rename_column("spoken_messages", "messages"),
            dailytalk.remove_columns("messages").rename_column("spoken_messages", "messages"),
            openhermes.remove_columns("messages").rename_column("spoken_messages", "messages"),
            everyday_conversations.remove_columns("messages").rename_column("speech_to_text_messages", "messages"),
            dailytalk.remove_columns("messages").rename_column("speech_to_text_messages", "messages"),
            openhermes.remove_columns("messages").rename_column("speech_to_text_messages", "messages"),
            everyday_conversations.remove_columns("messages").rename_column("text_to_speech_messages", "messages"),
            dailytalk.remove_columns("messages").rename_column("text_to_speech_messages", "messages"),
            openhermes.remove_columns("messages").rename_column("text_to_speech_messages", "messages"),
        ]
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_args.name, eos_token="<|im_end|>")
    chatml_token_ids = tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])

    def grad_hook(grad):
        masked_grad = torch.zeros_like(grad)
        masked_grad[chatml_token_ids] = grad[chatml_token_ids]
        return masked_grad

    # Model
    model = AutoModelForCausalLM.from_pretrained(config.model_args.name, eos_token_id=151645)
    model.model.layers.requires_grad_(False)
    model.model.norm.requires_grad_(False)
    handle_input_embeddings = model.get_input_embeddings().weight.register_hook(grad_hook)
    handle_output_embeddings = model.get_output_embeddings().weight.register_hook(grad_hook)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
