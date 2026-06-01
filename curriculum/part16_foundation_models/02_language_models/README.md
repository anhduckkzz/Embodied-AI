# Module 16.2 - Language Models

Code: [`dl/tokenizer.py`](../../../dl/tokenizer.py),
[`dl/transformer.py`](../../../dl/transformer.py) (the `GPT` class), demo in
[Module 16.1](../01_attention_transformers/build_gpt.py).

A language model assigns probabilities to sequences of tokens and, in practice,
predicts the next token given the previous ones. It is the language half of a
VLM, and in many VLAs the language model is the backbone that also produces
actions. This module explains how one is built and trained.

## 1. From text to tokens

A model operates on integer token ids, not raw text. Tokenization defines the
vocabulary and the mapping text to ids (Module reference:
[`dl/tokenizer.py`](../../../dl/tokenizer.py)).

- Character level: tiny vocabulary, long sequences, used in our demo.
- Subword / Byte Pair Encoding (BPE): the production standard (GPT, BERT). Start
  from characters and greedily merge the most frequent adjacent pairs into new
  tokens. This balances vocabulary size against sequence length and gracefully
  handles any input, including rare or unseen words. Tools: Hugging Face
  tokenizers, tiktoken, SentencePiece.

Each token id indexes a row of an embedding table, turning the id into a learnable
vector. Position information is added so the model knows token order (Module 16.1).

## 2. The training objective: next-token prediction

A decoder-only language model (GPT) is trained by self-supervision: predict the
next token given all previous tokens, using a causal mask so it cannot peek
ahead. The loss is cross-entropy between the predicted distribution and the actual
next token, averaged over every position. No human labels are needed, only text,
which is why these models can be trained on enormous corpora. See the loss in the
[`GPT`](../../../dl/transformer.py) forward method and the runnable
[`build_gpt.py`](../01_attention_transformers/build_gpt.py).

## 3. Generation

At inference the model predicts a distribution over the next token, samples one,
appends it, and repeats (autoregressive decoding). Temperature controls
randomness: low temperature is more deterministic, high is more diverse. See
[`GPT.generate`](../../../dl/transformer.py).

## 4. From a base model to an assistant

Pretraining on next-token prediction yields broad knowledge but not obedience.
Two further stages make a usable assistant, and both are concepts you already
met:

- Instruction tuning: supervised fine-tuning on instruction-response pairs. This
  is imitation learning (Part 12) applied to text.
- Preference optimization (RLHF / DPO): align behavior with human preferences
  (Part 1, Module 14). RLHF uses the very PPO you implemented.

This is the same imitate-then-refine recipe used for robot policies and VLAs.

## 5. Encoder vs decoder language models

- Decoder-only (GPT): causal, generative, predicts the next token. The dominant
  design and the backbone of most LLMs and VLAs.
- Encoder-only (BERT): bidirectional, builds representations for understanding
  tasks, trained by masked-token prediction.
- Encoder-decoder (T5): an encoder reads the input, a decoder generates the
  output; natural for translation and some multimodal setups.

## 6. Why this matters for robotics

In a VLA the language model provides two things: the ability to understand an
instruction ("put the red block in the bowl") and, in tokenized-action designs, a
ready-made autoregressive decoder that can emit action tokens the same way it
emits words (Part 10, Module 16.4, 16.5). The semantic knowledge from text
pretraining is what lets a robot generalize to objects and phrasings it never saw
in robot data.

## Project
1. Train [`build_gpt.py`](../01_attention_transformers/build_gpt.py) on a larger
   text file of your choice. Compare samples at temperature 0.5 versus 1.2.
2. Implement a tiny BPE: count adjacent pair frequencies in a corpus, merge the
   top pair repeatedly, and compare resulting sequence lengths to character level.
3. Explain, by tracing the code, why the causal mask is essential for the
   next-token objective to be well posed.

## Check your understanding
1. Why use subword (BPE) tokenization instead of words or characters?
2. What is the training objective of a decoder-only language model, and why does
   it need no labels?
3. What does temperature control during generation?
4. How do instruction tuning and RLHF relate to Parts 12 and 1?
5. Why is a pretrained language model useful inside a robot policy?

## Go deeper
- Radford et al. (2019), GPT-2; Brown et al. (2020), GPT-3 (in-context learning).
- Devlin et al. (2018), BERT; Raffel et al. (2020), T5.
- Karpathy, Let's build the GPT tokenizer (BPE from scratch).

Next: [Module 16.3 - Vision Models](../03_vision_models/).
