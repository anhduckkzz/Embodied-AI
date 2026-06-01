"""Build and train a small character-level GPT from scratch.

Run:  python curriculum/part16_foundation_models/01_attention_transformers/build_gpt.py

This uses dl/transformer.py (which you can read in full) to train a tiny language
model on a short text, then generates new characters. It is the same architecture
as a large language model, scaled down to run on a CPU in well under a minute. It
demonstrates the complete loop: tokenize, embed, attend through transformer
blocks, predict the next token, optimize the cross-entropy loss, and sample.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from dl.tokenizer import CharTokenizer
from dl.transformer import GPT

TEXT = (
    "the transformer reads a sequence of tokens. "
    "attention lets each token look at the others. "
    "stacking attention builds a language model. "
    "a language model predicts the next token. "
)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training a small GPT on {len(TEXT)} characters (device: {device})\n")

    tok = CharTokenizer(TEXT)
    data = tok.encode_tensor(TEXT)
    block_size = 32
    model = GPT(tok.vocab_size, dim=128, depth=3, num_heads=4, max_len=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print(f"vocabulary size: {tok.vocab_size} characters | "
          f"parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    for step in range(1, 1201):
        x, y = get_batch(data, block_size, batch_size=16, device=device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"step {step:>4}  loss {loss.item():.3f}")

    # Generate by continuing a prompt.
    prompt = "attention"
    idx = tok.encode_tensor(prompt).unsqueeze(0).to(device)
    out = model.generate(idx, max_new_tokens=120, temperature=0.8)
    print("\nGenerated continuation of 'attention':\n")
    print("  " + tok.decode(out[0].tolist()).replace("\n", " "))
    print("\nThe model learned the text's statistics from scratch. Scale this up "
          "with more data, depth, and compute and you get a real language model.")


if __name__ == "__main__":
    main()
