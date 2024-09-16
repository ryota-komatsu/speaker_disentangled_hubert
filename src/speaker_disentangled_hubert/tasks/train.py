from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.byol import BYOL
from ..models.dino import DINO
from ..utils.data import LibriSpeech
from ..utils.misc import fix_random_seed, get_tri_stage_schedule


def train(config):
    fix_random_seed(config.common.seed)

    train_dataset = ConcatDataset(
        [
            LibriSpeech(
                root=config.dataset.root,
                url="train-clean-100",
                download=config.dataset.download,
                max_sample_size=config.dataset.max_sample_size,
            ),
            LibriSpeech(
                root=config.dataset.root,
                url="train-clean-360",
                download=config.dataset.download,
                max_sample_size=config.dataset.max_sample_size,
            ),
            LibriSpeech(
                root=config.dataset.root,
                url="train-other-500",
                download=config.dataset.download,
                max_sample_size=config.dataset.max_sample_size,
            ),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
    )

    if config.model.model_type == "byol":
        model = BYOL(
            model_name_or_path=config.model.model_name_or_path,
            init_last_layer=config.model.init_last_layer,
            head_out_size=config.model.head_out_size,
            head_hidden_size=config.model.head_hidden_size,
            ema_decay=config.model.ema_decay,
        ).cuda()
    elif config.model.model_type == "dino":
        model = DINO(
            model_name_or_path=config.model.model_name_or_path,
            init_last_layer=config.model.init_last_layer,
            head_out_size=config.model.head_out_size,
            head_hidden_size=config.model.head_hidden_size,
            head_bottleneck_size=config.model.head_bottleneck_size,
            teacher_temp=config.model.teacher_temp,
            student_temp=config.model.student_temp,
            center_momentum=config.model.center_momentum,
            ema_decay=config.model.ema_decay,
        ).cuda()
    else:
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )

    # learning rate scheduler
    assert config.optim.stage_ratio[0] + config.optim.stage_ratio[1] + config.optim.stage_ratio[2] == 1
    T_max = config.optim.epoch * len(train_loader)
    warmup_steps = int(T_max * config.optim.stage_ratio[0])
    hold_steps = int(T_max * config.optim.stage_ratio[1])
    decay_steps = T_max - warmup_steps - hold_steps
    lr_scheduler = get_tri_stage_schedule(
        optimizer,
        config.optim.lr,
        config.optim.lr_min,
        warmup_steps,
        hold_steps,
        decay_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=config.common.fp16)
    writer = SummaryWriter()

    last_epoch = 0
    step = 0

    # resume training
    if Path(config.path.checkpoint).is_file():
        ckpt = torch.load(config.path.checkpoint)

        last_epoch = ckpt["epoch"]
        step = ckpt["step"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

        print(f"load from {config.path.checkpoint}")
        del ckpt

    if step < warmup_steps:
        model.freeze_pretrained_modules()
    else:
        model.defrost_transformer_encoder()

    for epoch in range(last_epoch + 1, config.optim.epoch + 1):
        model.train()

        for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=config.common.disable_tqdm):
            with torch.cuda.amp.autocast(enabled=config.common.fp16):
                loss = model(
                    teacher_input_values=batch["waveform"].cuda(),
                    student_input_values=batch["perturbed_waveform"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                )
            scaler.scale(loss).backward()

            # gradient clipping
            if config.optim.max_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_norm)

            # update student
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            optimizer.zero_grad()

            # update teacher
            model.update_teacher()

            # update learning rate
            lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

            step += 1

            # tensorboard log
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/scale", scale, step)
            if config.optim.max_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)

            if step == warmup_steps:
                model.defrost_transformer_encoder()

        # save model
        ckpt = {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        Path(config.path.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, config.path.checkpoint)
