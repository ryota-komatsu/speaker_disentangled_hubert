import torch

from ..s5hubert import SylRegForSyllableDiscovery


def get_input_embeddings(model_name_or_path: str, freeze: bool = True) -> torch.nn.Embedding:
    model = SylRegForSyllableDiscovery.from_pretrained(model_name_or_path)
    embeddings = torch.zeros(model.config.vocab_size, model.quantizer1.shape[1])

    for idx, unit in enumerate(model.quantizer2):
        embeddings[unit] += model.quantizer1[idx]

    embeddings /= torch.bincount(model.quantizer2).unsqueeze(1)
    embeddings /= 10
    embeddings = torch.cat((embeddings, torch.zeros(1, model.quantizer1.shape[1])))

    return torch.nn.Embedding.from_pretrained(embeddings, freeze=freeze, padding_idx=model.config.vocab_size)
