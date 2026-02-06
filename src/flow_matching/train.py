from pathlib import Path

import jiwer
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    pipeline,
)

from ..bigvgan.bigvgan import BigVGan, BigVGanConfig
from .configs import FlowMatchingConfig
from .data import get_collate_fn
from .models import FlowMatchingModel
from .utils import get_input_embeddings

# register BigVGan
AutoConfig.register("bigvgan", BigVGanConfig)
AutoModel.register(BigVGanConfig, BigVGan)


class EvaluationCallback(TrainerCallback):
    def __init__(
        self,
        vocoder_model_name_or_path,
        asr_model_name_or_path,
        eval_dataset,
        data_collator,
    ):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.vocoder = AutoModel.from_pretrained(vocoder_model_name_or_path, device_map="cuda")
        asr = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr_model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="cuda",
        )
        self.processor = AutoProcessor.from_pretrained(asr_model_name_or_path)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=asr,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=dtype,
        )

        self.dataloader = torch.utils.data.DataLoader(eval_dataset, collate_fn=data_collator)

    @torch.inference_mode()
    def on_step_end(self, args, state, control, model: FlowMatchingModel, **kwargs):
        if state.global_step % args.eval_steps != 0 or not state.is_world_process_zero:
            return

        model.eval()
        hyps = []

        for batch in self.dataloader:
            spectrogram = model.sample(batch["input_ids"].to(model.device)).spectrogram
            hyp_wav = self.vocoder(spectrogram)
            hyp_wav = hyp_wav.cpu().squeeze(0).numpy()
            hyp = self.pipe(hyp_wav, generate_kwargs={"language": "english"}, return_timestamps=True)["text"]
            hyps.append(hyp)

        transcripts = [
            self.processor.tokenizer.normalize(transcript) for transcript in self.dataloader.dataset["transcript"]
        ]
        hyps = [self.processor.tokenizer.normalize(hyp) for hyp in hyps]

        cer = jiwer.cer(transcripts, hyps) * 100
        wer = jiwer.wer(transcripts, hyps) * 100

        pd.DataFrame([cer, wer], index=["CER", "WER"]).to_csv(
            Path(args.output_dir) / f"score_dev_{state.global_step}.csv"
        )

        model.train()


def train_dit(config):
    libritts = load_dataset(config.dataset.name, "LibriTTS-R", split="train", keep_in_memory=True)
    emilia = load_dataset(config.dataset.name, "emilia", split="train", keep_in_memory=True)
    yodas = load_dataset(config.dataset.name, "yodas", split="train")
    train_dataset = concatenate_datasets([libritts, emilia, yodas])
    # eval_dataset = load_dataset(config.dataset.name, "LibriTTS-R", split="dev", keep_in_memory=True)

    train_dataset = train_dataset.with_format("torch")
    # eval_dataset = eval_dataset.with_format("torch")

    model = FlowMatchingModel(FlowMatchingConfig(**OmegaConf.to_container(config.flow_matching.model_args)))
    model.set_input_embeddings(get_input_embeddings(config.speech2unit.model_name_or_path))

    training_args = TrainingArguments(**OmegaConf.to_container(config.flow_matching.training_args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=get_collate_fn(config.flow_matching.model_args.vocab_size),
        # callbacks=[
        #     EvaluationCallback(
        #         vocoder_model_name_or_path=config.vocoder.model_name_or_path,
        #         asr_model_name_or_path=config.asr.model_name_or_path,
        #         eval_dataset=eval_dataset,
        #         data_collator=get_collate_fn(config.flow_matching.model_args.vocab_size),
        #     )
        # ],
    )
    trainer.train(resume_from_checkpoint=config.flow_matching.training_args.resume_from_checkpoint)


def finetune_dit(config):
    train_dataset = load_dataset(config.dataset.name, "Hi-Fi-CAPTAIN", split="female", keep_in_memory=True)
    train_dataset = train_dataset.with_format("torch")

    model = FlowMatchingModel.from_pretrained(config.flow_matching.finetuning_args.resume_from_checkpoint)

    training_args = TrainingArguments(**OmegaConf.to_container(config.flow_matching.finetuning_args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=get_collate_fn(config.flow_matching.model_args.vocab_size),
    )
    trainer.train()
