import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from transformers import Trainer, TrainerCallback, TrainingArguments

from ...sdhubert.utils.syllable import BoundaryDetectionEvaluator
from ..models.s5hubert import S5Hubert, S5HubertForSelfSegmentation
from ..models.s5hubert_dino import S5HubertDino
from ..utils.data import LibriLight, LibriSpeech
from ..utils.mincut import parallel_mincut


class EMACallback(TrainerCallback):
    def on_step_end(self, args, state, control, model, **kwargs):
        model.update_teacher()


class TeacherUpdateCallback(TrainerCallback):
    def __init__(self, teacher_update_steps):
        self.teacher_update_steps = teacher_update_steps

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step in self.teacher_update_steps:
            model.update_teacher()


class DefrostCallback(TrainerCallback):
    def __init__(self, defrost_steps: int = 2000):
        self.defrost_steps = defrost_steps

    def on_train_begin(self, args, state, control, model, **kwargs):
        if state.global_step < self.defrost_steps:
            model.freeze_pretrained_modules()

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step == self.defrost_steps:
            model.defrost_transformer_encoder()


class SavingCallback(TrainerCallback):
    def on_train_end(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero:
            model.student.save_pretrained(args.output_dir)


class EvaluationCallback(TrainerCallback):
    def __init__(self, config):
        self.config = config

    @torch.inference_mode()
    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % args.eval_steps != 0 or not state.is_world_process_zero:
            return

        model.eval()
        torch.cuda.empty_cache()

        wav_dir = Path(self.config.dataset.root) / "LibriSpeech"
        segment_dir = Path(self.config.path.segment_dir)
        segment_paths = []

        total_seconds = 0

        with open(self.config.dataset.dev_file) as f:
            for n, wav_name in enumerate(f):
                wav_name = wav_name.rstrip()
                wav_path = wav_dir / wav_name
                wav_path = str(wav_path)  # for sox backend
                wav, sr = torchaudio.load(wav_path)
                wav = wav.cuda()

                hidden_states, _ = model.student_forward(wav)
                hidden_states = hidden_states[self.config.model.segmentation_layer].squeeze(0).cpu().numpy()
                outputs = {"hidden_states": hidden_states}

                # save hidden states
                segment_name = wav_name.replace(".flac", ".npy")
                segment_path = segment_dir / segment_name
                segment_path.parent.mkdir(parents=True, exist_ok=True)
                segment_paths.append(segment_path)
                np.save(segment_path, outputs)

                total_seconds += wav.shape[1] / sr

        parallel_mincut(
            segment_paths,
            self.config.common.disable_tqdm,
            self.config.mincut.sec_per_frame,
            self.config.mincut.sec_per_syllable,
            self.config.mincut.merge_threshold,
            self.config.mincut.min_duration,
            self.config.mincut.max_duration,
            self.config.mincut.num_workers,
        )

        results = BoundaryDetectionEvaluator(
            self.config.path.segment_dir,
            self.config.dataset.dev_alignment,
            self.config.dataset.dev_alignment,
            tolerance=0.05,
            max_val_num=None,
        ).evaluate()

        # calculate the unit frequency
        num_units = 0

        for segment_path in segment_paths:
            ckpt = np.load(segment_path, allow_pickle=True)[()]
            num_units += len(ckpt["segments"])

        results["unit_frequency"] = num_units / total_seconds

        def round_float(results, ndigits: int = 3):
            if isinstance(results, dict):
                return {k: round_float(v) for k, v in results.items()}
            elif isinstance(results, float):
                return round(results, ndigits)
            else:
                return results

        results = round_float(results)

        with open(Path(args.output_dir) / f"score_dev_{state.global_step}.csv", "w") as f:
            json.dump(results, f, indent=2)

        model.train()
        torch.cuda.empty_cache()


def train(config):
    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args))

    train_dataset = LibriLight(
        data_dir=config.dataset.lh_dir,
        max_sample_size=config.dataset.max_sample_size,
        perturb=config.dataset.perturb,
    )

    if config.model.model_type == "s5hubert":
        model = S5Hubert(
            model_name_or_path=config.model.model_name_or_path,
            init_last_layer=config.model.init_last_layer,
            head_out_size=config.model.head_out_size,
            head_hidden_size=config.model.head_hidden_size,
            ema_decay=config.model.ema_decay,
        )
    elif config.model.model_type == "s5hubert_dino":
        model = S5HubertDino(
            model_name_or_path=config.model.model_name_or_path,
            init_last_layer=config.model.init_last_layer,
            head_out_size=config.model.head_out_size,
            head_hidden_size=config.model.head_hidden_size,
            head_bottleneck_size=config.model.head_bottleneck_size,
            teacher_temp=config.model.teacher_temp,
            student_temp=config.model.student_temp,
            center_momentum=config.model.center_momentum,
            ema_decay=config.model.ema_decay,
        )
    else:
        return

    model.defrost_transformer_encoder()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=LibriLight.collate_fn,
        callbacks=[EMACallback(), DefrostCallback(), SavingCallback(), EvaluationCallback(config)],
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)


def finetune(config):
    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args))
    model = S5HubertForSelfSegmentation(config.model.model_name_or_path)
    train_dataset = LibriSpeech(root=config.dataset.root, max_sample_size=None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=LibriSpeech.collate_fn2,
        callbacks=[TeacherUpdateCallback(config.model.teacher_update_steps), SavingCallback()],
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)
