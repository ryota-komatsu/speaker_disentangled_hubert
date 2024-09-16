import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from ..utils.data import LibriSpeech
from ..utils.misc import compute_mutual_info

sys.path.append("src/sdhubert")
from ...sdhubert.utils.misc import load_model


def compute_spk_normalized_mutual_info(p_xy: np.ndarray):
    mi = compute_mutual_info(p_xy)
    p_spk = np.sum(p_xy, axis=1)
    return mi / np.sum(-p_spk * np.log(p_spk))


@torch.inference_mode()
def analyze_sdhubert(config):
    test_dataset = torch.utils.data.ConcatDataset(
        [
            LibriSpeech(
                root=config.dataset.root,
                url="test-clean",
                download=config.dataset.download,
                max_sample_size=None,
            ),
            LibriSpeech(
                root=config.dataset.root,
                url="test-other",
                download=config.dataset.download,
                max_sample_size=None,
            ),
        ]
    )
    test_loader = torch.utils.data.DataLoader(test_dataset)

    sdhubert = load_model(config.path.checkpoint)[0]
    sdhubert.cuda()
    sdhubert.eval()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    spk_list = list()
    with open(Path(config.dataset.root) / "LibriSpeech/SPEAKERS.TXT") as f:
        for line in f:
            if line[0] == ";":
                continue

            spk_info = line.split("|")
            spk_id = int(spk_info[0].strip())
            split = spk_info[2].strip()

            if "test" in split:
                spk_list.append(spk_id)

    coocurrence_mat = torch.zeros(len(spk_list), 2048)

    for batch in tqdm(test_loader):
        waveform = batch["waveform"]
        spk_id = batch["spk_id"][0]

        waveform = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_values
        waveform = waveform.cuda()

        outputs = sdhubert(wav=waveform, inference_mode=True)
        pred_category = torch.argmax(outputs["cls"][0])

        coocurrence_mat[spk_list.index(spk_id), pred_category] += 1

    p_xy = coocurrence_mat.numpy()
    p_xy = p_xy / np.sum(p_xy)
    mi = compute_spk_normalized_mutual_info(p_xy)

    print(f"Speaker normalized mutual info.: {mi}")
