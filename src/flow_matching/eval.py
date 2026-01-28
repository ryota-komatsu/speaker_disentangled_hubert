import sys
import warnings
from pathlib import Path

import jiwer
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, HubertModel, pipeline

from ..s5hubert import S5HubertForSyllableDiscovery
from .models import FlowMatchingWithBigVGan

sys.path.append("src/utmos")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from ..utmos.score import Score

sys.path.append("src/textlesslib")
from src.textlesslib.textless import dispatch_quantizer
from src.textlesslib.textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder


def len_filter(example):
    return 4.0 <= len(example["audio"]["array"]) / example["audio"]["sampling_rate"] <= 10.0


def get_eval_fn(encoder, decoder, processor, pipe, scorer, data_dir):
    data_dir = Path(data_dir).resolve()

    def _evaluate(example):
        # transcript
        id = Path(example["audio"]["path"]).relative_to(data_dir).with_suffix("")
        speaker_id, chap_id, utterance_id = str(id).split("/")
        file = data_dir / speaker_id / chap_id / f"{speaker_id}-{chap_id}.trans.txt"

        with open(file) as f:
            for line in f:
                id, transcript = line.rstrip().split(" ", maxsplit=1)
                if id == utterance_id:
                    break

        example["transcript"] = processor.tokenizer.normalize(transcript)

        ref_wav = example["audio"]["array"].unsqueeze(0).to(encoder.device)
        hyp_wav = decoder(encoder(ref_wav)[0]["units"].unsqueeze(0)).waveform

        ref = pipe(ref_wav.cpu().squeeze(0).numpy(), generate_kwargs={"language": "english"}, return_timestamps=True)
        hyp = pipe(hyp_wav.cpu().squeeze(0).numpy(), generate_kwargs={"language": "english"}, return_timestamps=True)

        example["ref"] = processor.tokenizer.normalize(ref["text"])
        example["hyp"] = processor.tokenizer.normalize(hyp["text"])
        example["mos_ref"] = scorer.score(ref_wav)
        example["mos_hyp"] = scorer.score(hyp_wav)

        return example

    return _evaluate


@torch.inference_mode()
def evaluate(config):
    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")
    decoder = FlowMatchingWithBigVGan.from_pretrained(config.unit2speech.model_name_or_path, device_map="cuda")

    # model = HubertModel.from_pretrained("slprl/mhubert-base-25hz", num_hidden_layers=11, device_map="cuda")
    # quantizer = dispatch_quantizer("mhubert-base-25hz", "kmeans", 500)
    # vocoder = CodeHiFiGANVocoder.by_name(
    #     dense_model_name="mhubert-base-25hz",
    #     quantizer_model_name="kmeans",
    #     vocab_size=500,
    # ).cuda()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.asr.model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(config.asr.model_name_or_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
    )

    scorer = Score(ckpt_path="src/utmos/epoch=3-step=7459.ckpt", input_sample_rate=16000, device="cuda")

    data_dir = config.dataset.eval_dir
    dataset = load_dataset("audiofolder", data_dir=data_dir, split="train")
    dataset = dataset.filter(len_filter)
    dataset = dataset.with_format("torch")
    dataset = dataset.map(get_eval_fn(encoder, decoder, processor, pipe, scorer, data_dir))

    wer_hyp = jiwer.wer(dataset["transcript"], dataset["hyp"]) * 100
    cer_hyp = jiwer.cer(dataset["transcript"], dataset["hyp"]) * 100
    mos_hyp = torch.mean(dataset["mos_hyp"]).item()

    wer_ref = jiwer.wer(dataset["transcript"], dataset["ref"]) * 100
    cer_ref = jiwer.cer(dataset["transcript"], dataset["ref"]) * 100
    mos_ref = torch.mean(dataset["mos_ref"]).item()

    Path(config.path.result).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [wer_hyp, cer_hyp, mos_hyp, wer_ref, cer_ref, mos_ref],
        index=["WER (hyp)", "CER (hyp)", "MOS (hyp)", "WER (ref)", "CER (ref)", "MOS (ref)"],
    ).to_csv(config.path.result, float_format="%.2f")
