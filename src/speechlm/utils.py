# from https://github.com/slp-rl/slamkit/blob/main/slamkit/utils/calculation_utils.py

# MIT License
#
# Copyright (c) 2025 SLP-RL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import NLTKWordTokenizer
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import OPTConfig, PreTrainedTokenizerFast


def calc_ngram(text: str, nltk_word_tokenizer: NLTKWordTokenizer, n: int):
    tokens = nltk_word_tokenizer.tokenize(text)
    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return ngrams


def calc_auto_bleu(text: str, nltk_word_tokenizer: NLTKWordTokenizer, n: int):
    res = 0
    ngrams = calc_ngram(text, nltk_word_tokenizer, n)
    if len(ngrams) == 0:
        return 0
    for i in range(len(ngrams)):
        left = ngrams[:i]
        right = ngrams[i + 1 :]
        if ngrams[i] in left or ngrams[i] in right:
            res += 1
    return res / len(ngrams)


class OPTForSpeechLMConfig(OPTConfig):
    vocab_size: int = 8193
    hidden_size: int = 768
    num_hidden_layers: int = 12
    ffn_dim: int = 3072
    max_position_embeddings: int = 256
    dropout: float = 0.1
    num_attention_heads: int = 12
    activation_function: str = "gelu"
    pad_token_id: int = 8192
    bos_token_id: int | None = None
    eos_token_id: int = 8192


class SpeechLMTokenizerFast(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_size: int = 8192,
        bos_token: str | None = None,
        eos_token: str = "<|end_of_text|>",
        unk_token: str = "<|unk|>",
    ):
        vocab = [f"<{unit}>" for unit in range(vocab_size)] + [eos_token, unk_token]
        vocab = {token: token_id for token_id, token in enumerate(vocab)}

        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token=unk_token))
        tokenizer_object.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(r"<\d+>"), behavior="isolated")

        super().__init__(
            tokenizer_object=tokenizer_object,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=eos_token,
        )


def plot_results():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    c = np.logspace(21, 23, num=100)
    scaling_law = 26 * c**0.017

    plt.figure()

    plt.plot(c, scaling_law, ":", label="Scaling law [Cuervo+, EMNLP'24]")
    plt.scatter([2.1 * 10**21], [55.3], s=35**2, alpha=0.2, c="c")
    plt.scatter([5.4 * 10**22], [62.4], s=45**2, alpha=0.2, c="c")
    plt.scatter([4.0 * 10**21], [61.0], s=35**2, alpha=0.2, c="c")
    plt.scatter([2.9 * 10**22], [60.8], s=35**2, alpha=0.2, c="c")
    plt.scatter([1.3 * 10**21], [67.1], s=35**2, alpha=0.2, c="red")

    plt.annotate("TWIST", (0.65 * 2.1 * 10**21, 55.3), fontsize=14)
    plt.annotate("GLM-4-Voice", (0.65 * 5.4 * 10**22, 62.4), fontsize=14)
    plt.annotate("SpiRit-LM", (0.65 * 4.0 * 10**21, 61.0), fontsize=14)
    plt.annotate("Moshi", (0.65 * 2.9 * 10**22, 60.8), fontsize=14)
    plt.annotate("SylReg-LM", (0.65 * 1.3 * 10**21, 67.1), fontsize=14)

    plt.xlabel("Training compute (FLOPs)", fontsize=16)
    plt.ylabel("Spoken StoryCloze (%)", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xscale("log")
    plt.ylim(54, 68)
    plt.legend(fontsize=12)

    plt.savefig("docs/figures/results.png", bbox_inches="tight")
