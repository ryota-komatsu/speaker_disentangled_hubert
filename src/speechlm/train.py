import deepspeed
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from deepspeed.utils.tensor_fragment import fragment_address
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .data.utils import get_collator
from .trainer import SpeechLMTrainer

torch.serialization.add_safe_globals(
    [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.UInt32DType,
        deepspeed.runtime.fp16.loss_scaler.LossScaler,
        deepspeed.runtime.zero.config.ZeroStageEnum,
        fragment_address,
    ]
)


def train(config):
    deepspeed.init_distributed()

    # initialize `TrainingArguments` *before* instantiating your model for ``is_deepspeed_zero3_enabled`` query
    # https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/training_args.py#L714
    # https://github.com/huggingface/accelerate/blob/v1.13.0/src/accelerate/utils/deepspeed.py#L163
    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_args.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.get_vocab()
    vocab_size = config.speech2unit.vocab_size
    units = [f"<{unit}>" for unit in range(vocab_size)]
    for unit in units:
        assert unit not in vocab
    tokenizer.add_tokens(units)

    # Datasets
    librilight = load_dataset(config.dataset.name, "Libri-Light", split="train", keep_in_memory=True)
    libriheavy = load_dataset(config.dataset.name, "libriheavy", split="train", keep_in_memory=True)
    librispeech = load_dataset(config.dataset.name, "LibriSpeech", split="train", keep_in_memory=True)
    tinystories = load_dataset(config.dataset.name, "TinyStories", split="train", keep_in_memory=True)
    peoples_speech = load_dataset(config.dataset.name, "peoples_speech", split="train", keep_in_memory=True)
    voxpopuli = load_dataset(config.dataset.name, "voxpopuli", split="train", keep_in_memory=True)

    train_dataset = concatenate_datasets(
        [
            libriheavy,
            libriheavy,
            librispeech,
            librispeech,
            tinystories,
            tinystories,
            peoples_speech,
            peoples_speech,
            voxpopuli,
            voxpopuli,
            librilight,
            librilight,
            librispeech.remove_columns("aligned_units"),
            librispeech.remove_columns("aligned_units"),
            tinystories.remove_columns("aligned_units"),
            tinystories.remove_columns("aligned_units"),
            peoples_speech.remove_columns("aligned_units"),
            peoples_speech.remove_columns("aligned_units"),
            voxpopuli.remove_columns("aligned_units"),
            voxpopuli.remove_columns("aligned_units"),
        ]
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(config.model_args.name)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=config.model_args.mean_resizing)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=get_collator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)
