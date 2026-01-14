import os
from pathlib import Path

import fire
from omegaconf import OmegaConf


class TaskRunner:
    def tokenize_eval(self, config: str = "configs/speechlm/default.yaml"):
        from src.speechlm.data.utils import tokenize_eval

        config = OmegaConf.load(config)
        tokenize_eval(config)

    def train(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())
        # os.environ["HF_DATASETS_OFFLINE"] = "1"
        # os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from src.speechlm.train import train

        train(config)

    def evaluate(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())

        from src.speechlm.eval import evaluate

        evaluate(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
