import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from ..models.hubert import HubertForSyllableDiscovery
from ..models.s5hubert import S5HubertForSyllableDiscovery
from ..models.sylreg import SylRegForSyllableDiscovery
from ..models.vghubert import VGHubertForSyllableDiscovery
from ..utils.data import LibriSpeech
from ..utils.mincut import parallel_mincut

MODELS = {
    "hubert": HubertForSyllableDiscovery,
    "vghubert": VGHubertForSyllableDiscovery,
}


def _syllable_segmentation(config):
    if config.model.model_type.startswith("s5hubert"):
        model = S5HubertForSyllableDiscovery.from_pretrained(
            config.path.checkpoint,
            segmentation_layer=config.model.segmentation_layer,
            sec_per_syllable=config.mincut.sec_per_syllable,
            merge_threshold=config.mincut.merge_threshold,
            min_duration=config.mincut.min_duration,
            max_duration=config.mincut.max_duration,
        ).cuda()
    elif config.model.model_type in MODELS:
        model = MODELS[config.model.model_type](
            checkpoint_path=config.path.checkpoint,
            quantizer1_path=None,
            quantizer2_path=None,
            segmentation_layer=config.model.segmentation_layer,
        ).cuda()
    elif config.model.model_type == "sylboost":
        sys.path.append("src/SyllableLM")
        from ...SyllableLM.extract_units import SylBoostFeatureReader

        model = SylBoostFeatureReader(
            config.path.checkpoint,
            config.path.quantizer1,
            config.path.quantizer2,
            config.model.model_key,
        )
    elif config.model.model_type == "sylber":
        from ...sylber.sylber import Segmenter

        model = Segmenter(config.path.checkpoint)
    else:
        return

    wav_dir = Path(config.dataset.root) / "LibriSpeech"
    segment_dir = Path(config.path.segment_dir)
    segment_paths = []
    files = [
        config.dataset.train_file,
        config.dataset.dev_file,
        config.dataset.test_file,
    ]

    for file in files:
        with open(file) as f:
            lines = f.readlines()
            for wav_name in tqdm(lines, disable=config.common.disable_tqdm):
                wav_name = wav_name.rstrip()
                wav_path = wav_dir / wav_name
                wav_path = str(wav_path)  # for sox backend
                wav, sr = torchaudio.load(wav_path)

                if config.model.model_type.startswith("s5hubert") or config.model.model_type in MODELS:
                    wav = wav.cuda()
                    hidden_states = model.get_hidden_states(wav).cpu().numpy()
                    outputs = {"hidden_states": hidden_states}
                elif config.model.model_type == "sylboost":
                    wav = wav.cuda()
                    outputs = model.forward(wav)
                    outputs = {
                        "segments": outputs["clusters_with_times"][0][1:].T * config.mincut.sec_per_frame,
                        "units": outputs["clusters_with_times"][0][0],
                    }
                elif config.model.model_type == "sylber":
                    wav = wav.squeeze(0).numpy()
                    outputs = model(wav=wav)

                # save hidden states
                segment_name = wav_name.replace(".flac", ".npy")
                segment_path = segment_dir / segment_name
                segment_path.parent.mkdir(parents=True, exist_ok=True)
                segment_paths.append(segment_path)
                np.save(segment_path, outputs)

    if config.model.model_type.startswith("s5hubert") or config.model.model_type in MODELS:
        parallel_mincut(
            segment_paths,
            config.common.disable_tqdm,
            config.mincut.sec_per_frame,
            config.mincut.sec_per_syllable,
            config.mincut.merge_threshold,
            config.mincut.min_duration,
            config.mincut.max_duration,
            config.mincut.num_workers,
        )


def syllable_segmentation(config):
    if config.model.model_type.startswith("s5hubert"):
        model = S5HubertForSyllableDiscovery.from_pretrained(
            config.path.checkpoint,
            segmentation_layer=config.model.segmentation_layer,
            sec_per_syllable=config.mincut.sec_per_syllable,
            merge_threshold=config.mincut.merge_threshold,
            min_duration=config.mincut.min_duration,
            max_duration=config.mincut.max_duration,
        ).cuda()
    else:
        model = SylRegForSyllableDiscovery.from_pretrained(config.path.checkpoint).cuda()

    dataset = ConcatDataset(
        [
            LibriSpeech(root=config.dataset.root, url="train-clean-100", max_sample_size=None, perturb=False),
            LibriSpeech(root=config.dataset.root, url="train-clean-360", max_sample_size=None, perturb=False),
            LibriSpeech(root=config.dataset.root, url="train-other-500", max_sample_size=None, perturb=False),
            LibriSpeech(root=config.dataset.root, url="dev-clean", max_sample_size=None, perturb=False),
            LibriSpeech(root=config.dataset.root, url="dev-other", max_sample_size=None, perturb=False),
            LibriSpeech(root=config.dataset.root, url="test-clean", max_sample_size=None, perturb=False),
            LibriSpeech(root=config.dataset.root, url="test-other", max_sample_size=None, perturb=False),
        ]
    )
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=LibriSpeech.collate_fn)

    segment_dir = Path(config.path.segment_dir)

    for batch in tqdm(dataloader):
        batch_outputs = model(
            batch["teacher_input_values"].cuda(),
            batch["teacher_attention_mask"].cuda(),
        )

        # save hidden states
        for wav_name, outputs in zip(batch["wav_names"], batch_outputs):
            segment_name = wav_name.replace(".flac", ".npy")
            segment_path = segment_dir / segment_name
            segment_path.parent.mkdir(parents=True, exist_ok=True)
            outputs = {
                "segments": outputs["segments"].cpu().numpy(),
                "segment_features": outputs["segment_features"].cpu().numpy(),
            }
            np.save(segment_path, outputs)
