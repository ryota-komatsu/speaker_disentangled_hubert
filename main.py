import argparse

from omegaconf import OmegaConf

from src.speaker_disentangled_hubert.tasks.clustering import clustering
from src.speaker_disentangled_hubert.tasks.eval import evaluate
from src.speaker_disentangled_hubert.tasks.syllable_segmentation import syllable_segmentation
from src.speaker_disentangled_hubert.tasks.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    train(config)
    syllable_segmentation(config)
    clustering(config)
    evaluate(config)
