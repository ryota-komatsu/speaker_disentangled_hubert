import fire
from omegaconf import OmegaConf

from src.s5hubert.tasks.eval import evaluate
from src.s5hubert.tasks.layerwise_analysis import layerwise_analysis
from src.s5hubert.tasks.quantize import quantize
from src.s5hubert.tasks.speaker_identification import speaker_identification
from src.s5hubert.tasks.syllable_segmentation import syllable_segmentation
from src.s5hubert.tasks.train import finetune, train


class TaskRunner:
    def train(self, config: str = "configs/speech2unit/default.yaml"):
        config = OmegaConf.load(config)
        train(config)

    def finetune(self, config: str = "configs/speech2unit/default.yaml"):
        config = OmegaConf.load(config)
        finetune(config)

    def syllable_segmentation(self, config: str = "configs/speech2unit/default.yaml"):
        config = OmegaConf.load(config)
        syllable_segmentation(config)

    def quantize(self, config: str = "configs/speech2unit/default.yaml"):
        config = OmegaConf.load(config)
        quantize(config)

    def evaluate(self, config: str = "configs/speech2unit/default.yaml"):
        config = OmegaConf.load(config)
        evaluate(config)

    def speaker_identification(self, config: str = "configs/speaker_identification/default.yaml"):
        config = OmegaConf.load(config)
        speaker_identification(config)

    def layerwise_analysis(self, config: str = "configs/speech2unit/default.yaml"):
        config = OmegaConf.load(config)
        layerwise_analysis(config)

    def __call__(self, config: str = "configs/speech2unit/default.yaml"):
        config = OmegaConf.load(config)
        train(config)
        syllable_segmentation(config)
        quantize(config)
        evaluate(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
