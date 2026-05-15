import sys

import torch
from torch.utils.data import ConcatDataset
from torchaudio.functional import edit_distance
from tqdm import tqdm

from ..models.s5hubert import S5HubertForSyllableDiscovery
from ..utils.data import LibriSpeech


def calc_ued(config):
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

    print(total_ued / total_len * 100)
