from pathlib import Path

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
    def __init__(
        self,
        swuggy_dev_loader: torch.utils.data.DataLoader,
        sblimp_dev_loader: torch.utils.data.DataLoader,
        swuggy_test_loader: torch.utils.data.DataLoader,
        sblimp_test_loader: torch.utils.data.DataLoader,
        tSC_test_loader: torch.utils.data.DataLoader,
    ):
        self.swuggy_dev_loader = swuggy_dev_loader
        self.sblimp_dev_loader = sblimp_dev_loader
        self.swuggy_test_loader = swuggy_test_loader
        self.sblimp_test_loader = sblimp_test_loader
        self.tSC_test_loader = tSC_test_loader

    @torch.inference_mode()
    def _eval(
        self,
        model,
        loader: torch.utils.data.DataLoader,
    ):
        for batch in loader:
            # Speech LM
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            logits = model(input_ids=input_ids, labels=labels).logits.transpose(1, 2)

            labels = F.pad(labels, (0, 1), value=-100)
            shifted_labels = labels[:, 1:]

            scores = -F.cross_entropy(logits, shifted_labels, reduction="none")
            scores = scores.sum(dim=1) / scores.ne(0).sum(dim=1)
            scores = scores.tolist()

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % args.eval_steps != 0 or not state.is_world_process_zero:
            return

        model.eval()

        self._eval(model, self.swuggy_dev_loader)
        self._eval(model, self.sblimp_dev_loader)

        # pd.DataFrame(
        #    np.array([swuggy_all, swuggy_iv, swuggy_oov, sblimp]) * 100,
        #    index=["sWUGGY", "sWUGGY iv", "sWUGGY oov", "sBLIMP"],
        # ).to_csv(Path(args.output_dir) / f"scores/score_dev_{state.global_step}.csv")

        model.train()

    def on_train_end(self, args, state, control, model, **kwargs):
        if not state.is_world_process_zero:
            return

        model.eval()

        self._eval(model, self.swuggy_test_loader)
        self._eval(model, self.sblimp_test_loader)
        self._eval(model, self.tSC_test_loader)

        # pd.DataFrame(
        #    np.array([swuggy_all, swuggy_iv, swuggy_oov, sblimp, tSC]) * 100,
        #    index=["sWUGGY", "sWUGGY iv", "sWUGGY oov", "sBLIMP", "tSC"],
        # ).to_csv(Path(args.output_dir) / "scores/score_test.csv")


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
    swuggy = load_dataset(config.dataset.name, "sWUGGY")
    sblimp = load_dataset(config.dataset.name, "sBLIMP")
    tSC = load_dataset(config.dataset.name, "tSC")

    swuggy_dev_loader = torch.utils.data.DataLoader(
        swuggy["dev"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collator(tokenizer),
    )
    sblimp_dev_loader = torch.utils.data.DataLoader(
        sblimp["dev"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collator(tokenizer),
    )
    swuggy_test_loader = torch.utils.data.DataLoader(
        swuggy["test"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collator(tokenizer),
    )
    sblimp_test_loader = torch.utils.data.DataLoader(
        sblimp["test"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collator(tokenizer),
    )
    tSC_test_loader = torch.utils.data.DataLoader(
        tSC["test"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collator(tokenizer),
    )

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
            EvaluationCallback(
                swuggy_dev_loader,
                sblimp_dev_loader,
                swuggy_test_loader,
                sblimp_test_loader,
                tSC_test_loader,
            ),
            DefrostCallback(handle_input_embeddings, handle_output_embeddings),
        ],
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)
