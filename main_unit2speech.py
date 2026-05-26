import os
from pathlib import Path

import fire
from omegaconf import OmegaConf


class TaskRunner:
    def tokenize(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())

        from src.flow_matching.data import tokenize

        tokenize(config)

    def train_bigvgan(self, config: str = "configs/unit2speech/default.yaml"):
        from src.bigvgan.train import train_bigvgan

        config = OmegaConf.load(config)
        train_bigvgan(config)

    def train_dit(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())

        from src.flow_matching.train import train_dit

        train_dit(config)

    def finetune_dit(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())

        from src.flow_matching.train import finetune_dit

        finetune_dit(config)

    def evaluate(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())

        from src.flow_matching.eval import evaluate

        evaluate(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
