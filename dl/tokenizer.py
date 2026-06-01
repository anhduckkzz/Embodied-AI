"""Tokenization: turning text into integers a model can consume.

A model does not see characters or words, it sees integer ids that index an
embedding table. A tokenizer defines the vocabulary and the mapping
``text <-> ids``. The choice of vocabulary matters:

    Character level (this file): tiny vocabulary, very long sequences, no
        unknown-word problem. Perfect for learning and small experiments.
    Word level: short sequences but a huge vocabulary and trouble with rare or
        unseen words.
    Subword / BPE (Byte Pair Encoding): the practical standard used by GPT, BERT,
        and the language side of VLMs. It starts from characters and greedily
        merges the most frequent adjacent pairs into new tokens, balancing
        vocabulary size against sequence length and handling any input. Real
        implementations: Hugging Face ``tokenizers``, ``tiktoken``,
        SentencePiece.

We implement the character-level tokenizer in full so the concept is concrete,
and explain BPE in comments so you know what the production tools do.
"""
from __future__ import annotations

from typing import List

import torch


class CharTokenizer:
    """Map each unique character in a corpus to an integer id, and back.

    This is the simplest possible tokenizer. The same interface (encode/decode,
    a fixed ``vocab_size``) is what a BPE tokenizer exposes, just with subword
    tokens instead of single characters.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def encode_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)
