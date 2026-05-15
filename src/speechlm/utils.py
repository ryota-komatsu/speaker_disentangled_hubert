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
    def __init__(
        self,
        vocab_size: int = 8193,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        ffn_dim: int = 3072,
        max_position_embeddings: int = 256,
        dropout: float = 0.1,
        num_attention_heads: int = 12,
        activation_function="gelu",
        pad_token_id: int = 8192,
        bos_token_id: int = None,
        eos_token_id: int = 8192,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
            activation_function=activation_function,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class SpeechLMTokenizerFast(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_size: int = 8192,
        bos_token: str = None,
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
