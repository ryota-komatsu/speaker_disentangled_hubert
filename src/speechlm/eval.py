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


class Evaluator:
    def __init__(
        self,
        speech2unit_model_name_or_path: str,
        unit2speech_model_name_or_path: str,
        speechlm_model_name_or_path: str,
        textlm_model_name_or_path: str,
        asr_model_name_or_path: str,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.nltk_word_tokenizer = NLTKWordTokenizer()

        self.encoder = SylRegForSyllableDiscovery.from_pretrained(
            speech2unit_model_name_or_path,
            device_map=device,
            dtype="auto",
        )
        self.decoder = FlowMatchingWithBigVGan.from_pretrained(
            unit2speech_model_name_or_path,
            device_map=device,
            dtype="auto",
        )

        self.speechlm = AutoModelForCausalLM.from_pretrained(
            speechlm_model_name_or_path,
            device_map=device,
            dtype="auto",
        )
        self.speechlm_tokenizer = AutoTokenizer.from_pretrained(speechlm_model_name_or_path)

        self.textlm = AutoModelForCausalLM.from_pretrained(textlm_model_name_or_path, device_map=device, dtype="auto")
        self.textlm_tokenizer = AutoTokenizer.from_pretrained(textlm_model_name_or_path)

        asr = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr_model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(asr_model_name_or_path)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=asr,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
        )

    @torch.inference_mode()
    def evaluate_understanding(self, batch: dict[str, list]):
        pos_units = ["".join(f"<{unit}>" for unit in pair["pos"]) for pair in batch["units"]]
        neg_units = ["".join(f"<{unit}>" for unit in pair["neg"]) for pair in batch["units"]]
        units = pos_units + neg_units

        inputs = self.speechlm_tokenizer(units, padding=True, return_tensors="pt").to(self.speechlm.device)

        logits = self.speechlm(**inputs).logits.transpose(1, 2)

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

    @torch.inference_mode()
    def evaluate_generation(
        self,
        example,
        prompt_length: int = 3,
        generation_length: int = 10,
        do_sample: bool = True,
        temperature: float = 0.8,
        auto_bleu_n: int = 2,
    ):
        # 1. encode a waveform into syllabic units
        prompt = example["audio"]["array"][: prompt_length * example["audio"]["sampling_rate"]]
        prompt = prompt.unsqueeze(0).to(self.encoder.device)
        outputs = self.encoder(prompt)
        units = outputs[0]["units"][:-1]  # [3950, 67, ..., 503]

        # 2. speech language modeling
        input_text = "".join(f"<{unit}>" for unit in units)
        input_ids = self.speechlm_tokenizer(input_text, padding=True, return_tensors="pt").input_ids.to(
            self.speechlm.device
        )
        generated_ids = self.speechlm.generate(input_ids=input_ids, do_sample=do_sample, temperature=temperature)[0]
        units = self.speechlm_tokenizer.decode(generated_ids)
        units = torch.tensor([int(unit) for unit in re.findall(r"<(\d+)>", units)], device=self.decoder.device)

        # 3. unit-to-speech synthesis
        generated_speech = self.decoder(units.unsqueeze(0)).waveform
        generated_speech = generated_speech[:, : generation_length * 16000]
        generated_speech = generated_speech.cpu().squeeze(0).numpy()

        # 4. ASR
        generated_text = self.pipe(generated_speech, generate_kwargs={"language": "english"}, return_timestamps=True)[
            "text"
        ]

        # 5. negative log-likelihood
        input_ids = self.textlm_tokenizer(generated_text, padding=True, return_tensors="pt").input_ids.to(
            self.textlm.device
        )
        nll = self.textlm(input_ids=input_ids, labels=input_ids).loss.cpu().item()

        example["nll"] = nll
        example["auto-bleu"] = calc_auto_bleu(generated_text, self.nltk_word_tokenizer, auto_bleu_n)

        return example


def evaluate(config):
    # 1. load models
    evaluator = Evaluator(
        config.speech2unit.model_name_or_path,
        config.unit2speech.model_name_or_path,
        config.training_args.resume_from_checkpoint,
        config.textlm.model_name_or_path,
        config.asr.model_name_or_path,
    )

    # 2. load datasets
    eval_dataset = {
        "sWUGGY": load_dataset(config.dataset.name, "sWUGGY"),
        "sBLIMP": load_dataset(config.dataset.name, "sBLIMP"),
        "tSC": load_dataset(config.dataset.name, "tSC"),
        "sSC": load_dataset(config.dataset.name, "sSC"),
        "SALMon_sentiment_alignment": load_dataset(config.dataset.name, "SALMon_sentiment_alignment"),
        "generation": load_dataset("audiofolder", data_dir=config.dataset.eval_dir).with_format("torch"),
    }

    map_kwargs = dict(batched=True, batch_size=config.training_args.per_device_eval_batch_size)

    # 3. evaluate
    sWUGGY = eval_dataset["sWUGGY"]["test"].map(evaluator.evaluate_understanding, **map_kwargs)
    sBLIMP = eval_dataset["sBLIMP"]["test"].map(evaluator.evaluate_understanding, **map_kwargs)
    tSC = eval_dataset["tSC"]["test"].map(evaluator.evaluate_understanding, **map_kwargs)
    sSC = eval_dataset["sSC"]["test"].map(evaluator.evaluate_understanding, **map_kwargs)
    SALMon_sentiment_alignment = eval_dataset["SALMon_sentiment_alignment"]["test"].map(
        evaluator.evaluate_understanding, **map_kwargs
    )
    generation = eval_dataset["generation"]["train"].map(evaluator.evaluate_generation)

    # 4. save results
    results = {
        "sWUGGY": np.mean(sWUGGY["metrics"]),
        "sBLIMP": np.mean(sBLIMP["metrics"]),
        "tSC": np.mean(tSC["metrics"]),
        "sSC": np.mean(sSC["metrics"]),
        "SALMon_sentiment_alignment": np.mean(SALMon_sentiment_alignment["metrics"]),
        "perplexity": np.float64(generation["nll"].mean().exp().item()),
        "auto-bleu": np.float64(generation["auto-bleu"].mean().item()),
    }
    Path(config.training_args.output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(results, orient="index").to_csv(Path(config.training_args.output_dir) / "score_test.csv")
