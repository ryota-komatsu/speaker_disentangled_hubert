from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.byol import BYOLForSequenceClassification
from ..models.hubert import HubertForSequenceClassification
from ..models.sdhubert import SDHubertForSequenceClassification
from ..models.vghubert import VGHubertForSequenceClassification
from ..utils.data import VoxCeleb
from ..utils.misc import fix_random_seed

MODELS = {
    "byol": BYOLForSequenceClassification,
    "hubert": HubertForSequenceClassification,
    "sdhubert": SDHubertForSequenceClassification,
    "vghubert": VGHubertForSequenceClassification,
}


def evaluate(config, data_loader, model, writer: Optional[SummaryWriter] = None, epoch: Optional[int] = None):
    model.eval()
    accuracy = 0

    with torch.cuda.amp.autocast(enabled=config.common.fp16):
        for batch in tqdm(data_loader, disable=config.common.disable_tqdm):
            outputs = model(
                input_values=batch["waveform"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                labels=batch["labels"].cuda(),
            )
            pred_label = torch.argmax(outputs.logits.squeeze(0))
            accuracy += pred_label == batch["labels"].item()

    accuracy = accuracy / len(data_loader)
    if writer:
        writer.add_scalar(f"dev/accuracy", accuracy, epoch)
    return accuracy


def speaker_identification(config):
    fix_random_seed(config.common.seed)

    train_set = VoxCeleb(
        root=config.dataset.root,
        subset="train",
        download=config.dataset.download,
        max_sample_size=config.dataset.max_sample_size,
    )
    dev_set = VoxCeleb(
        root=config.dataset.root,
        subset="dev",
        download=config.dataset.download,
        max_sample_size=None,
    )
    test_set = VoxCeleb(
        root=config.dataset.root,
        subset="test",
        download=config.dataset.download,
        max_sample_size=None,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        num_workers=config.dataloader.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        num_workers=config.dataloader.num_workers,
    )

    model = MODELS[config.model.model_type](
        model_name_or_path=config.model.model_name_or_path,
        classifier_proj_size=config.model.classifier_proj_size,
        num_labels=config.model.num_labels,
        segmentation_layer=config.model.segmentation_layer,
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=config.common.fp16)
    writer = SummaryWriter()

    last_epoch = 0
    step = 0
    best_dev_accuracy = 0

    if Path(config.path.checkpoint).is_file():
        ckpt = torch.load(config.path.checkpoint)

        last_epoch = ckpt["epoch"]
        step = ckpt["step"]
        best_dev_accuracy = ckpt["dev_accuracy"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

        print(f"load from {config.path.checkpoint}")
        del ckpt

    for epoch in range(last_epoch + 1, config.optim.epoch + 1):
        model.train()
        model.hubert.eval()

        for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=config.common.disable_tqdm):
            with torch.cuda.amp.autocast(enabled=config.common.fp16):
                loss = model(
                    batch["waveform"].cuda(),
                    batch["attention_mask"].cuda(),
                    labels=batch["labels"].cuda(),
                ).loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_norm)

            # update
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            optimizer.zero_grad()

            step += 1

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/grad_norm", grad_norm, step)
            writer.add_scalar("train/scale", scale, step)

        dev_accuracy = evaluate(config, dev_loader, model, writer, epoch)

        ckpt = {
            "epoch": epoch,
            "step": step,
            "dev_accuracy": dev_accuracy,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }
        Path(config.path.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        if best_dev_accuracy < dev_accuracy:
            best_dev_accuracy = dev_accuracy
            torch.save(ckpt, config.path.checkpoint)

    # test the best model
    ckpt = torch.load(config.path.checkpoint)
    model.load_state_dict(ckpt["model"])
    test_accuracy = evaluate(config, test_loader, model)
    print(f"test accuracy: {test_accuracy}")
