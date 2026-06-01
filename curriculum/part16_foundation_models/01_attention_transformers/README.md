# Module 16.1 - Attention and Transformers

Code: [`dl/attention.py`](../../../dl/attention.py),
[`dl/transformer.py`](../../../dl/transformer.py). Demo:
[`build_gpt.py`](build_gpt.py).

This is the foundation of the entire modern model stack. Language models, vision
models, vision-language models, and vision-language-action models are all built
from the transformer block. If you understand this module deeply, every higher
layer becomes a variation on a theme you already know.

## 1. The problem attention solves

Earlier sequence models (RNNs, LSTMs) process tokens one at a time and must
squeeze all past context into a single hidden state. They struggle with long
range dependencies and are slow to train because they are inherently sequential.

Attention replaces recurrence with direct, content based lookup. Every position
can look at every other position in one step, in parallel. This is both more
expressive (no information bottleneck) and far more efficient on modern hardware,
which is why it scales to the enormous models we use today.

## 2. Attention, step by step

For each position we form three vectors by linear projection of its features:

- Query (Q): what this position is looking for.
- Key (K): what each position offers for matching.
- Value (V): the content each position contributes if selected.

The computation, exactly as in [`scaled_dot_product_attention`](../../../dl/attention.py):

```
scores  = Q K^T / sqrt(d)      # match every query against every key
weights = softmax(scores)      # normalize into attention weights (sum to 1)
output  = weights V            # weighted average of the values
```

The scaling by `1/sqrt(d)` keeps the dot products from growing with dimension,
which would otherwise push softmax into tiny gradients. The output for each
position is a blend of the values it found most relevant.

## 3. Self-attention vs cross-attention

- Self-attention: Q, K, V all come from the same sequence. The sequence relates
  to itself (a sentence understanding its own words). This is the default inside
  language and vision encoders.
- Cross-attention: Q comes from one sequence, K and V from another. One modality
  reads from another (text tokens attending to image patches). This is the
  fusion mechanism inside VLMs and VLAs, covered in Module 16.4.

Both are the same operation; only the source of Q versus K and V differs. See the
`context` argument in [`MultiHeadAttention`](../../../dl/attention.py).

## 4. Multi-head attention

A single attention pattern captures one type of relationship. Multi-head
attention runs several attention operations in parallel on different learned
projections, then concatenates them. Each head can specialize, for example one
tracking word order and another tracking subject to object links. We split the
model dimension across heads so the cost stays roughly constant.

## 5. Masking

A causal mask (lower triangular) prevents a position from attending to future
positions. This is what makes a language model autoregressive: when predicting
token t it may only use tokens 1..t. Encoders (BERT, the ViT body) use no causal
mask, so every position sees the whole input. See
[`causal_mask`](../../../dl/attention.py).

## 6. The transformer block

Stack attention with a small per-position feed-forward network, wrap each with a
residual connection and layer normalization (the modern pre-norm form):

```
x = x + attention(norm(x))      # mix information across positions
x = x + feedforward(norm(x))    # transform each position independently
```

Residual connections let gradients flow through deep stacks; layer norm keeps
activations well scaled. Stacking many of these blocks is the whole architecture.
See [`TransformerBlock`](../../../dl/transformer.py). Positional encodings are
added because attention itself ignores order (Module covers this in the code
comments of `PositionalEncoding`).

## 7. One block, three uses

| Configuration | Result | Example |
|---------------|--------|---------|
| causal self-attention | decoder / language model | GPT (Module 16.2) |
| bidirectional self-attention | encoder | BERT, ViT body (Module 16.3) |
| cross-attention | modality fusion | VLM, VLA (Module 16.4) |

## 8. Build it yourself

[`build_gpt.py`](build_gpt.py) trains a small character-level GPT from
[`dl/transformer.py`](../../../dl/transformer.py) on a short text and generates
new text. It is small enough to run on a CPU in under a minute, and it is the
same architecture, in miniature, as the language models that back VLAs.

```bash
python curriculum/part16_foundation_models/01_attention_transformers/build_gpt.py
```

## Project
1. Run `build_gpt.py`. Watch the loss fall and the generated text become more
   coherent. Increase `depth` and the training text, and compare.
2. Print and visualize the attention weights for a sample input. Identify which
   positions a given token attends to.
3. Implement the sinusoidal `PositionalEncoding` as an alternative to the learned
   positions in `GPT`, and confirm training still works.

## Check your understanding
1. What do Q, K, and V represent, and how do they combine to form the output?
2. Why divide the scores by sqrt(d)?
3. What is the only difference between self-attention and cross-attention?
4. Why does a language model need a causal mask but an encoder does not?
5. What do the residual connections and layer norm contribute to a deep stack?

## Go deeper
- Vaswani et al. (2017), Attention Is All You Need.
- Karpathy, Let's build GPT and the nanoGPT repository (build a transformer from
  scratch, the best companion to this module).
- The Illustrated Transformer (Jay Alammar).

Next: [Module 16.2 - Language Models](../02_language_models/).
