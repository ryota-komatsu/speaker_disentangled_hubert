import os
from pathlib import Path

import fire
from omegaconf import OmegaConf


class TaskRunner:
    def tokenize_librilight(
        self,
        num_shards: int = 1,
        shard_index: int = 0,
        data_dir: str = "data/librilight",
        model_name_or_path: str = "ryota-komatsu/s5-hubert",
        tgt_len_sec: int = 25,
        min_len_sec: int = 5,
        max_len_sec: int = 30,
    ):
        from src.speechlm.data.librilight import tokenize_librilight_

        tokenize_librilight_(
            num_shards,
            shard_index,
            data_dir,
            model_name_or_path,
            tgt_len_sec,
            min_len_sec,
            max_len_sec,
        )

    def tokenize_librispeech(
        self,
        data_dir: str = "data/LibriSpeech",
        model_name_or_path: str = "ryota-komatsu/s5-hubert",
    ):
        from src.speechlm.data.librispeech import tokenize_librispeech

        tokenize_librispeech(data_dir, model_name_or_path)

    def tokenize_peoples_speech_clean(
        self,
        num_shards: int = 1,
        shard_index: int = 0,
        data_dir: str = "data/peoples_speech",
        model_name_or_path: str = "ryota-komatsu/s5-hubert",
    ):
        from src.speechlm.data.peoples_speech import tokenize_clean

        tokenize_clean(num_shards, shard_index, data_dir, model_name_or_path)

    def tokenize_peoples_speech_clean_sa(
        self,
        data_dir: str = "data/peoples_speech",
        model_name_or_path: str = "ryota-komatsu/s5-hubert",
    ):
        from src.speechlm.data.peoples_speech import tokenize_clean_sa

        tokenize_clean_sa(data_dir, model_name_or_path)

    def tokenize_tinystories(
        self,
        num_shards: int = 1,
        shard_index: int = 0,
        data_dir: str = "data/tinystories",
        model_name_or_path: str = "ryota-komatsu/s5-hubert",
    ):
        from src.speechlm.data.tinystories import tokenize_tinystories

        tokenize_tinystories(num_shards, shard_index, data_dir, model_name_or_path)

    def tokenize_voxpopuli(
        self,
        data_dir: str = "data/voxpopuli",
        model_name_or_path: str = "ryota-komatsu/s5-hubert",
    ):
        from src.speechlm.data.voxpopuli import tokenize_voxpopuli

        tokenize_voxpopuli(data_dir, model_name_or_path)

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
