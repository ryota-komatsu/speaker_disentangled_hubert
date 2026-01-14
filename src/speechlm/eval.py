from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_evaluator(model, processing_class):
    @torch.inference_mode()
    def evaluator(batch: Dict[str, list]):
        pos_units = ["".join(f"<{unit}>" for unit in pair["pos"]) for pair in batch["units"]]
        neg_units = ["".join(f"<{unit}>" for unit in pair["neg"]) for pair in batch["units"]]
        units = pos_units + neg_units

        inputs = processing_class(units, padding=True, return_tensors="pt").to(model.device)

        logits = model(**inputs).logits.transpose(1, 2)

        labels = inputs.input_ids.masked_fill(inputs.attention_mask.bool().logical_not(), -100)
        labels = F.pad(labels, (0, 1), value=-100)
        labels = labels[:, 1:]

        # log likelihood
        scores = -F.cross_entropy(logits, labels, reduction="none")
        scores = scores.sum(dim=1) / labels.ne(-100).sum(dim=1)
        pos_scores, neg_scores = scores.chunk(2)

        metrics = torch.zeros_like(pos_scores)
        metrics[pos_scores > neg_scores] = 100
        metrics[pos_scores == neg_scores] = 50
        metrics[pos_scores < neg_scores] = 0

        batch["metrics"] = metrics.tolist()
        return batch

    return evaluator


def evaluate(config):
    model = AutoModelForCausalLM.from_pretrained(config.training_args.resume_from_checkpoint, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(config.training_args.resume_from_checkpoint)
    global_step = config.training_args.resume_from_checkpoint.rsplit("-", 1)[1]

    eval_dataset = {
        "sWUGGY": load_dataset(config.dataset.name, "sWUGGY"),
        "sBLIMP": load_dataset(config.dataset.name, "sBLIMP"),
        "tSC": load_dataset(config.dataset.name, "tSC"),
        "sSC": load_dataset(config.dataset.name, "sSC"),
    }

    map_kwargs = dict(batched=True, batch_size=config.training_args.per_device_eval_batch_size)

    sWUGGY = eval_dataset["sWUGGY"]["test"].map(get_evaluator(model, tokenizer), **map_kwargs)
    sBLIMP = eval_dataset["sBLIMP"]["test"].map(get_evaluator(model, tokenizer), **map_kwargs)
    tSC = eval_dataset["tSC"]["test"].map(get_evaluator(model, tokenizer), **map_kwargs)
    sSC = eval_dataset["sSC"]["test"].map(get_evaluator(model, tokenizer), **map_kwargs)

    def is_in_vocab(example):
        return example["frequency"] != 0

    def is_out_of_vocab(example):
        return example["frequency"] == 0

    pd.DataFrame(
        [
            np.mean(sWUGGY["metrics"]),
            np.mean(sWUGGY.filter(is_in_vocab)["metrics"]),
            np.mean(sWUGGY.filter(is_out_of_vocab)["metrics"]),
            np.mean(sBLIMP["metrics"]),
            np.mean(tSC["metrics"]),
            np.mean(sSC["metrics"]),
        ],
        index=["sWUGGY", "sWUGGY IV", "sWUGGY OOV", "sBLIMP", "tSC", "sSC"],
    ).to_csv(Path(config.training_args.output_dir) / f"score_test_{global_step}.csv")
