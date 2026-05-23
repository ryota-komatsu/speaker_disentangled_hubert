import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchaudio.functional import edit_distance
from tqdm import tqdm

from ..models.s5hubert import S5HubertForSyllableDiscovery
from ..models.sylreg import SylRegForSyllableDiscovery
from ..utils.data import LibriSpeech


def compute_syllable_purity(p_xy: np.ndarray):
    return np.sum(np.max(p_xy, axis=0))


def compute_cluster_purity(p_xy: np.ndarray):
    return np.sum(np.max(p_xy, axis=1))


def compute_mutual_info(p_xy: np.ndarray):
    n_syllables, n_clusters = p_xy.shape

    p_syllable = np.sum(p_xy, axis=1)
    p_cluster = np.sum(p_xy, axis=0)

    mi = 0
    for i in range(n_syllables):
        for j in range(n_clusters):
            if p_xy[i, j] != 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_syllable[i] * p_cluster[j]))

    return mi


def compute_ued(config) -> float:
    if config.model.model_type.startswith("s5hubert"):
        model = S5HubertForSyllableDiscovery.from_pretrained(config.path.checkpoint).cuda()
    elif config.model.model_type == "sylboost":
        sys.path.append("src/SyllableLM")
        from ...SyllableLM.extract_units import SylBoostFeatureReader

        model = SylBoostFeatureReader(
            config.path.checkpoint,
            config.path.quantizer1,
            config.path.quantizer2,
            config.model.model_key,
        )

    dataset = ConcatDataset(
        [
            LibriSpeech(root=config.dataset.root, url="test-clean", max_sample_size=None),
            LibriSpeech(root=config.dataset.root, url="test-other", max_sample_size=None),
        ]
    )

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=LibriSpeech.collate_fn)

    total_ued = 0
    total_len = 0

    for batch in tqdm(data_loader):
        # original
        if config.model.model_type.startswith("s5hubert"):
            refs = model(
                input_values=batch["teacher_input_values"].cuda(),
                attention_mask=batch["teacher_attention_mask"].cuda(),
            )
        elif config.model.model_type == "sylboost":
            outputs = model.forward(batch["teacher_input_values"].cuda())
            refs = [{"units": outputs["clusters_with_times"][0][0]}]

        # speaker perturbation
        if config.model.model_type.startswith("s5hubert"):
            hyps = model(
                input_values=batch["student_input_values"].cuda(),
                attention_mask=batch["student_attention_mask"].cuda(),
            )
        elif config.model.model_type == "sylboost":
            outputs = model.forward(batch["student_input_values"].cuda())
            hyps = [{"units": outputs["clusters_with_times"][0][0]}]

        # unit edit distance (UED)
        # https://arxiv.org/abs/2209.15483
        for ref, hyp in zip(refs, hyps):
            total_ued += edit_distance(ref["units"], hyp["units"])
            total_len += len(ref["units"])

    return total_ued / total_len * 100
