import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from nltk.tokenize import NLTKWordTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline

from ..flow_matching import FlowMatchingWithBigVGan
from ..s5hubert import SylRegForSyllableDiscovery
from .utils import calc_auto_bleu


def get_evaluator(model, processing_class):
    @torch.inference_mode()
    def evaluator(batch: dict[str, list]):
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


def get_generation_evaluator(
    encoder: SylRegForSyllableDiscovery,
    speechlm,
    speechlm_tokenizer,
    decoder: FlowMatchingWithBigVGan,
    pipe,
    textlm,
    textlm_tokenizer,
    nltk_word_tokenizer: NLTKWordTokenizer,
    prompt_length: int = 3,
    generation_length: int = 10,
    do_sample: bool = True,
    temperature: float = 0.8,
    auto_bleu_n: int = 2,
):
    @torch.inference_mode()
    def _evaluate(example):
        # 1. encode a waveform into syllabic units
        prompt = example["audio"]["array"][: prompt_length * example["audio"]["sampling_rate"]]
        prompt = prompt.unsqueeze(0).to(encoder.device)
        outputs = encoder(prompt)
        units = outputs[0]["units"][:-1]  # [3950, 67, ..., 503]

        # 2. speech language modeling
        input_text = "".join(f"<{unit}>" for unit in units)
        input_ids = speechlm_tokenizer(input_text, padding=True, return_tensors="pt").input_ids.to(speechlm.device)
        generated_ids = speechlm.generate(input_ids=input_ids, do_sample=do_sample, temperature=temperature)[0]
        units = speechlm_tokenizer.decode(generated_ids)
        units = torch.tensor([int(unit) for unit in re.findall(r"<(\d+)>", units)], device=decoder.device)

        # 3. unit-to-speech synthesis
        generated_speech = decoder(units.unsqueeze(0)).waveform
        generated_speech = generated_speech[:, : generation_length * 16000]
        generated_speech = generated_speech.cpu().squeeze(0).numpy()

        # 4. ASR
        generated_text = pipe(generated_speech, generate_kwargs={"language": "english"}, return_timestamps=True)["text"]

        # 5. negative log-likelihood
        input_ids = textlm_tokenizer(generated_text, padding=True, return_tensors="pt").input_ids.to(textlm.device)
        nll = textlm(input_ids=input_ids, labels=input_ids).loss.cpu().item()

        example["nll"] = nll
        example["auto-bleu"] = calc_auto_bleu(generated_text, nltk_word_tokenizer, auto_bleu_n)

        return example

    return _evaluate


def evaluate(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. load models
    nltk_word_tokenizer = NLTKWordTokenizer()

    encoder = SylRegForSyllableDiscovery.from_pretrained(
        config.speech2unit.model_name_or_path,
        device_map=device,
        dtype="auto",
    )
    decoder = FlowMatchingWithBigVGan.from_pretrained(
        config.unit2speech.model_name_or_path,
        device_map=device,
        dtype="auto",
    )

    speechlm = AutoModelForCausalLM.from_pretrained(
        config.training_args.resume_from_checkpoint,
        device_map=device,
        dtype="auto",
    )
    speechlm_tokenizer = AutoTokenizer.from_pretrained(config.training_args.resume_from_checkpoint)

    textlm = AutoModelForCausalLM.from_pretrained(config.textlm.model_name_or_path, device_map=device, dtype="auto")
    textlm_tokenizer = AutoTokenizer.from_pretrained(config.textlm.model_name_or_path)

    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.asr.model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(config.asr.model_name_or_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
    )

    # 2. load datasets
    eval_dataset = {
        "sWUGGY": load_dataset(config.dataset.name, "sWUGGY"),
        "sBLIMP": load_dataset(config.dataset.name, "sBLIMP"),
        "tSC": load_dataset(config.dataset.name, "tSC"),
        "sSC": load_dataset(config.dataset.name, "sSC"),
        "generation": load_dataset("audiofolder", data_dir=config.dataset.eval_dir).with_format("torch"),
    }

    map_kwargs = dict(batched=True, batch_size=config.training_args.per_device_eval_batch_size)

    # 3. evaluate
    sWUGGY = eval_dataset["sWUGGY"]["test"].map(get_evaluator(speechlm, speechlm_tokenizer), **map_kwargs)
    sBLIMP = eval_dataset["sBLIMP"]["test"].map(get_evaluator(speechlm, speechlm_tokenizer), **map_kwargs)
    tSC = eval_dataset["tSC"]["test"].map(get_evaluator(speechlm, speechlm_tokenizer), **map_kwargs)
    sSC = eval_dataset["sSC"]["test"].map(get_evaluator(speechlm, speechlm_tokenizer), **map_kwargs)
    generation = eval_dataset["generation"]["train"].map(
        get_generation_evaluator(
            encoder,
            speechlm,
            speechlm_tokenizer,
            decoder,
            pipe,
            textlm,
            textlm_tokenizer,
            nltk_word_tokenizer,
        )
    )

    # 4. save results
    results = {
        "sWUGGY": np.mean(sWUGGY["metrics"]),
        "sBLIMP": np.mean(sBLIMP["metrics"]),
        "tSC": np.mean(tSC["metrics"]),
        "sSC": np.mean(sSC["metrics"]),
        "perplexity": np.float64(generation["nll"].mean().exp().item()),
        "auto-bleu": np.float64(generation["auto-bleu"].mean().item()),
    }
    Path(config.training_args.output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(results, orient="index").to_csv(Path(config.training_args.output_dir) / "score_test.csv")
