from pathlib import Path

import numpy as np
import torchaudio
from tqdm import tqdm

from ..mincut.mincut_utils import parallel_mincut
from ..models.byol import BYOLForSyllableDiscovery
from ..models.dino import DINOForSyllableDiscovery
from ..models.hubert import HubertForSyllableDiscovery
from ..models.vghubert import VGHubertForSyllableDiscovery

MODELS = {
    "byol": BYOLForSyllableDiscovery,
    "dino": DINOForSyllableDiscovery,
    "hubert": HubertForSyllableDiscovery,
    "vghubert": VGHubertForSyllableDiscovery,
}


def syllable_segmentation(config):
    if config.model.model_type in MODELS:
        model = MODELS[config.model.model_type](
            checkpoint_path=config.path.checkpoint,
            quantizer1_path=None,
            quantizer2_path=None,
            segmentation_layer=config.model.segmentation_layer,
        ).cuda()
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
            for wav_name in tqdm(f, disable=config.common.disable_tqdm):
                wav_name = wav_name.rstrip()
                wav_path = wav_dir / wav_name
                wav, sr = torchaudio.load(wav_path)
                wav = wav.cuda()

                outputs = model(wav)

                # save hidden states
                segment_name = wav_name.replace(".flac", ".npy")
                segment_path = segment_dir / segment_name
                segment_path.parent.mkdir(parents=True, exist_ok=True)
                segment_paths.append(segment_path)
                np.save(segment_path, outputs)

    parallel_mincut(segment_paths, config.common.disable_tqdm)
