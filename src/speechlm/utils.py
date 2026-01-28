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
