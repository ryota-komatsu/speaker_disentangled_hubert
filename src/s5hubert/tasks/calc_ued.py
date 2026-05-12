import torch
from torch.utils.data import ConcatDataset
from torchaudio.functional import edit_distance
from tqdm import tqdm

from ..models.s5hubert import S5HubertForSyllableDiscovery
from ..utils.data import LibriSpeech


def calc_ued(config):
    if config.model.model_type.startswith("s5hubert"):
        model = S5HubertForSyllableDiscovery.load_pretrained(
            config.path.checkpoint,
            config.path.quantizer1,
            config.path.quantizer2,
        ).cuda()
    else:
        return

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
        refs = model(
            input_values=batch["teacher_input_values"].cuda(),
            attention_mask=batch["teacher_attention_mask"].cuda(),
        )

        # speaker perturbation
        hyps = model(
            input_values=batch["student_input_values"].cuda(),
            attention_mask=batch["student_attention_mask"].cuda(),
        )

        # unit edit distance (UED)
        # https://arxiv.org/abs/2209.15483
        for ref, hyp in zip(refs, hyps):
            total_ued += edit_distance(ref["units"], hyp["units"])
            total_len += len(ref["units"])

    print(total_ued / total_len * 100)
