import fire
from omegaconf import OmegaConf


class TaskRunner:
    def train(self, config: str = "configs/speech2unit/default.yaml"):
        from src.s5hubert.tasks.train import train

        config = OmegaConf.load(config)
        train(config)

    def finetune(self, config: str = "configs/speech2unit/default.yaml"):
        from src.s5hubert.tasks.train import finetune

        config = OmegaConf.load(config)
        finetune(config)

    def syllable_segmentation(
        self,
        config: str = "configs/speech2unit/default.yaml",
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        from src.s5hubert.tasks.syllable_segmentation import syllable_segmentation

        config = OmegaConf.load(config)
        syllable_segmentation(config, num_shards, shard_index)

    def quantize(self, config: str = "configs/speech2unit/default.yaml"):
        from src.s5hubert.tasks.quantize import quantize

        config = OmegaConf.load(config)
        quantize(config)

    def evaluate(self, config: str = "configs/speech2unit/default.yaml"):
        from src.s5hubert.tasks.eval import evaluate

        config = OmegaConf.load(config)
        evaluate(config)

    def layerwise_analysis(self, config: str = "configs/speech2unit/default.yaml"):
        from src.s5hubert.tasks.layerwise_analysis import layerwise_analysis

        config = OmegaConf.load(config)
        layerwise_analysis(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
