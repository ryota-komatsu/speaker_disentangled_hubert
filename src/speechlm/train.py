from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from .data import get_collator

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType])


class EvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset):
        self.eval_dataset = eval_dataset

    def get_evaluator(self, model, processing_class):
        @torch.inference_mode()
        def evaluator(batch: Dict[str, list]):
            pos_units = ["".join(f"<{unit}>" for unit in pair["pos"]) for pair in batch["units"]]
            neg_units = ["".join(f"<{unit}>" for unit in pair["neg"]) for pair in batch["units"]]
            units = pos_units + neg_units

            inputs = processing_class(units, padding=True, return_tensors="pt").to(model.device)

            logits = model(**inputs).logits.transpose(1, 2)

            labels = inputs.input_ids.masked_fill(inputs.attention_mask.bool().logical_not(), -100)
            labels = F.pad(labels, (0, 1), value=-100)
            shifted_labels = labels[:, 1:]

            # log likelihood
            scores = -F.cross_entropy(logits, shifted_labels, reduction="none")
            scores = scores.sum(dim=1) / shifted_labels.ne(-100).sum(dim=1)
            pos_scores, neg_scores = scores.chunk(2)

            metrics = torch.zeros_like(pos_scores)
            metrics[pos_scores > neg_scores] = 100
            metrics[pos_scores == neg_scores] = 50
            metrics[pos_scores < neg_scores] = 0

            batch["metrics"] = metrics.cpu().numpy()
            return batch

        return evaluator

    def on_step_end(self, args, state, control, model, processing_class, **kwargs):
        if state.global_step % args.eval_steps != 0 or not state.is_world_process_zero:
            return

        model.eval()

        map_kwargs = dict(batched=True, batch_size=args.per_device_eval_batch_size)

        sWUGGY = self.eval_dataset["sWUGGY"]["validation"].map(
            self.get_evaluator(model, processing_class), **map_kwargs
        )
        sBLIMP = self.eval_dataset["sBLIMP"]["validation"].map(
            self.get_evaluator(model, processing_class), **map_kwargs
        )

        def is_in_vocab(example):
            return example["frequency"] != 0

        def is_out_of_vocab(example):
            return example["frequency"] == 0

        pd.DataFrame(
            [
                sWUGGY["metrics"].mean(),
                sWUGGY.filter(is_in_vocab)["metrics"].mean(),
                sWUGGY.filter(is_out_of_vocab)["metrics"].mean(),
                sBLIMP["metrics"].mean(),
            ],
            index=["sWUGGY", "sWUGGY IV", "sWUGGY OOV", "sBLIMP"],
        ).to_csv(Path(args.output_dir) / f"score_dev_{state.global_step}.csv")

        model.train()

    def on_train_end(self, args, state, control, model, processing_class, **kwargs):
        if not state.is_world_process_zero:
            return

        model.eval()

        map_kwargs = dict(batched=True, batch_size=args.per_device_eval_batch_size)

        sWUGGY = self.eval_dataset["sWUGGY"]["test"].map(self.get_evaluator(model, processing_class), **map_kwargs)
        sBLIMP = self.eval_dataset["sBLIMP"]["test"].map(self.get_evaluator(model, processing_class), **map_kwargs)
        tSC = self.eval_dataset["tSC"]["test"].map(self.get_evaluator(model, processing_class), **map_kwargs)

        def is_in_vocab(example):
            return example["frequency"] != 0

        def is_out_of_vocab(example):
            return example["frequency"] == 0

        pd.DataFrame(
            [
                sWUGGY["metrics"].mean(),
                sWUGGY.filter(is_in_vocab)["metrics"].mean(),
                sWUGGY.filter(is_out_of_vocab)["metrics"].mean(),
                sBLIMP["metrics"].mean(),
                tSC["metrics"].mean(),
            ],
            index=["sWUGGY", "sWUGGY IV", "sWUGGY OOV", "sBLIMP", "tSC"],
        ).to_csv(Path(args.output_dir) / f"score_test_{state.global_step}.csv")


class DefrostCallback(TrainerCallback):
    def __init__(self, handle_input_embeddings, handle_output_embeddings):
        self.handle_input_embeddings = handle_input_embeddings
        self.handle_output_embeddings = handle_output_embeddings

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step == args.warmup_steps:
            self.handle_input_embeddings.remove()
            self.handle_output_embeddings.remove()
            model.requires_grad_(True)


def train(config):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    vocab = tokenizer.get_vocab()
    vocab_size = config.speech2unit.vocab_size
    units = [f"<{unit}>" for unit in range(vocab_size)]
    for unit in units:
        assert unit not in vocab
    tokenizer.add_tokens(units)

    # Datasets
    train_dataset = load_dataset(config.dataset.name, "Libri-Light", split="train", keep_in_memory=True)
    eval_dataset = {
        "sWUGGY": load_dataset(config.dataset.name, "sWUGGY"),
        "sBLIMP": load_dataset(config.dataset.name, "sBLIMP"),
        "tSC": load_dataset(config.dataset.name, "tSC"),
    }

    # Model
    model = AutoModelForCausalLM.from_pretrained(config.model.name)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=config.model.mean_resizing)
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)
    model.get_output_embeddings().requires_grad_(True)
    handle_input_embeddings = model.get_input_embeddings().weight.register_hook(
        lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
    )
    handle_output_embeddings = model.get_output_embeddings().weight.register_hook(
        lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
    )

    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=get_collator(tokenizer),
        callbacks=[
            EvaluationCallback(eval_dataset),
            DefrostCallback(handle_input_embeddings, handle_output_embeddings),
        ],
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)
