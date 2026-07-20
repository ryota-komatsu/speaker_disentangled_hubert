import deepspeed
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from deepspeed.utils.tensor_fragment import fragment_address
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments

from .data.cosmopedia import filter_fn
from .data.utils import get_collator
from .trainer import SpeechLMTrainer
from .utils import OPTForSpeechLMConfig, SpeechLMTokenizerFast

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
    training_args = TrainingArguments(
        output_dir="models/SylReg-LM-7B-Stage1",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=8,
        learning_rate=0.0003,
        weight_decay=0.0,  # During this stage, if weight_decay > 0.0, text performance significantly degrades
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=0.5,
        max_steps=5000,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=None,
        bf16=True,
        use_liger_kernel=True,
        eval_steps=5000,
        dataloader_num_workers=16,
        remove_unused_columns=False,
        deepspeed="configs/speechlm/ds_zero2_bf16.json",
        optim="adamw_torch_fused",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        resume_from_checkpoint=None,
    )

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

    # Generation config
    speech_token_ids = tokenizer.convert_tokens_to_ids(units)
    speech_token_ids = set(speech_token_ids + [tokenizer.eos_token_id])
    bad_words_ids = [[token_id] for token_id in range(len(tokenizer)) if token_id not in speech_token_ids]
    config = GenerationConfig(max_length=128, do_sample=True, temperature=0.8, bad_words_ids=bad_words_ids)
    # config.push_to_hub("ryota-komatsu/SylReg-LM-7B")

    # Datasets
    librilight = load_dataset(config.dataset.name, "Libri-Light", split="train", keep_in_memory=True, num_proc=6)
    libriheavy = load_dataset(config.dataset.name, "libriheavy", split="train", keep_in_memory=True, num_proc=6)
    librispeech = load_dataset(config.dataset.name, "LibriSpeech", split="train", keep_in_memory=True, num_proc=6)
    tinystories = load_dataset(config.dataset.name, "TinyStories", split="train", keep_in_memory=True, num_proc=6)
    peoples_speech = load_dataset(config.dataset.name, "peoples_speech", split="train", keep_in_memory=True, num_proc=6)
    voxpopuli = load_dataset(config.dataset.name, "voxpopuli", split="train", keep_in_memory=True, num_proc=6)
    emilia = load_dataset(config.dataset.name, "emilia", split="train", keep_in_memory=True, num_proc=6)
    emilia_yodas = load_dataset(config.dataset.name, "emilia_yodas", split="train", keep_in_memory=True, num_proc=6)
    cosmopedia_speech = load_dataset(config.dataset.name, "cosmopedia-v2", split="train", num_proc=6)

    cosmopedia = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", num_proc=6)
    cosmopedia = cosmopedia.filter(filter_fn, num_proc=64)

    train_dataset = concatenate_datasets(
        [
            # speech-text interleaving
            libriheavy,
            librispeech,
            tinystories,
            peoples_speech,
            voxpopuli,
            emilia,
            emilia_yodas,
            cosmopedia_speech,
            # speech-only
            librilight,
            librispeech.remove_columns("aligned_units"),
            tinystories.remove_columns("aligned_units"),
            peoples_speech.remove_columns("aligned_units"),
            voxpopuli.remove_columns("aligned_units"),
            emilia.remove_columns("aligned_units"),
            emilia_yodas.remove_columns("aligned_units"),
            cosmopedia_speech.remove_columns("aligned_units"),
        ]
    )

    # Model
    # During the first 5k steps, we update only the randomly initialized syllabic token embeddings on speech-only and interleaved data
    # See Table 6 in https://openreview.net/forum?id=zjaV5zmlkl
    model = AutoModelForCausalLM.from_pretrained(config.model_args.name)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.model.layers.requires_grad_(False)
    model.model.norm.requires_grad_(False)
    handle_input_embeddings = model.get_input_embeddings().weight.register_hook(
        lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
    )
    handle_output_embeddings = model.get_output_embeddings().weight.register_hook(
        lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=get_collator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)

    training_args = TrainingArguments(
        output_dir="models/SylReg-LM-7B-Stage2",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=16,
        learning_rate=0.0003,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=0.5,
        max_steps=10000,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={"num_decay_steps": 5000, "min_lr_ratio": 0.1, "decay_type": "linear"},
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=None,
        bf16=True,
        use_liger_kernel=True,
        eval_steps=5000,
        dataloader_num_workers=16,
        remove_unused_columns=False,
        deepspeed="configs/speechlm/ds_zero2_bf16.json",
        optim="adamw_torch_fused",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        resume_from_checkpoint=None,
    )

    # Datasets
    # Mix speech-only, text-only, and interleaved examples in a roughly 3:3:4 ratio
    train_dataset = concatenate_datasets(
        [
            # text-only
            cosmopedia.select_columns("text"),
            # speech-text interleaving
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
            emilia,
            emilia,
            emilia_yodas,
            emilia_yodas,
            cosmopedia_speech,
            cosmopedia_speech,
            # speech-only
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
            emilia.remove_columns("aligned_units"),
            emilia.remove_columns("aligned_units"),
            emilia_yodas.remove_columns("aligned_units"),
            emilia_yodas.remove_columns("aligned_units"),
            cosmopedia_speech.remove_columns("aligned_units"),
            cosmopedia_speech.remove_columns("aligned_units"),
        ]
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained("models/SylReg-LM-7B-Stage1/checkpoint-5000")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=get_collator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)
