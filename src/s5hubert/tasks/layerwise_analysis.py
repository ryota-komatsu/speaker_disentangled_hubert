from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import HubertModel

from ...sdhubert.utils.syllable import BoundaryDetectionEvaluator
from ..utils.mincut import parallel_mincut

plt.rcParams["text.usetex"] = True
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


@torch.inference_mode()
def layerwise_analysis(config):
    model = HubertModel.from_pretrained(config.path.checkpoint).cuda().eval()

    wav_dir = Path(config.dataset.root) / "LibriSpeech"
    segment_dir = Path(config.path.segment_dir)
    segment_paths = [[] for _ in range(model.config.num_hidden_layers)]

    total_seconds = 0
    df = []

    with open(config.dataset.dev_file) as f:
        for n, wav_name in enumerate(f):
            wav_name = wav_name.rstrip()
            wav_path = wav_dir / wav_name
            wav_path = str(wav_path)  # for sox backend
            wav, sr = torchaudio.load(wav_path)
            wav = wav.cuda()

            hidden_states = model(wav, output_hidden_states=True).hidden_states
            hidden_states = hidden_states[1:]

            for segmentation_layer in range(model.config.num_hidden_layers):
                _hidden_states = hidden_states[segmentation_layer].squeeze(0).cpu().numpy()
                outputs = {"hidden_states": _hidden_states}

                # save hidden states
                segment_name = wav_name.replace(".flac", ".npy")
                segment_path = segment_dir / f"{segmentation_layer}" / segment_name
                segment_path.parent.mkdir(parents=True, exist_ok=True)
                segment_paths[segmentation_layer].append(segment_path)
                np.save(segment_path, outputs)

            total_seconds += wav.shape[1] / sr

    for segmentation_layer in range(model.config.num_hidden_layers):
        parallel_mincut(
            segment_paths[segmentation_layer],
            config.common.disable_tqdm,
            config.mincut.sec_per_frame,
            config.mincut.sec_per_syllable,
            config.mincut.merge_threshold,
            config.mincut.min_duration,
            config.mincut.max_duration,
            config.mincut.num_workers,
        )

        results = BoundaryDetectionEvaluator(
            segment_dir / f"{segmentation_layer}",
            config.dataset.dev_alignment,
            config.dataset.dev_alignment,
            tolerance=0.05,
            max_val_num=None,
        ).evaluate()

        # calculate the unit frequency
        num_units = 0

        for segment_path in segment_paths[segmentation_layer]:
            ckpt = np.load(segment_path, allow_pickle=True)[()]
            num_units += len(ckpt["segments"])

        results["unit_frequency"] = num_units / total_seconds

        results = pd.DataFrame.from_dict([results])
        df.append(results)

    result_dir = Path(config.path.result).parent
    result_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(df)
    df.to_csv(result_dir / "layer-wise.csv")

    plt.figure()
    plt.plot(range(1, 1 + model.config.num_hidden_layers), 100 * df["prec"], "v--", label="Precision")
    plt.plot(range(1, 1 + model.config.num_hidden_layers), 100 * df["recall"], "^--", label="Recall")
    plt.plot(range(1, 1 + model.config.num_hidden_layers), 100 * df["f1"], "o--", label="F1")
    plt.plot(range(1, 1 + model.config.num_hidden_layers), 100 * df["r_val"], "*--", label="R-value")
    plt.grid()
    plt.legend(fontsize=14)
    plt.xticks(range(1, 1 + model.config.num_hidden_layers), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"Transformer encoder layer $l$", fontsize=16)
    plt.ylabel("Segmentation scores", fontsize=16)
    plt.savefig(result_dir / "layer-wise.pdf", bbox_inches="tight")
