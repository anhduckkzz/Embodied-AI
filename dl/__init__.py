"""dl: a from-scratch, readable deep learning library for understanding model
architecture from the bottom up.

This is the third teaching library in the curriculum, alongside ``rl`` (decision
making) and ``robotics`` (classical robotics). Its job is to demystify the model
stack that modern robotics and autonomous driving are built on:

    attention -> transformer -> language model
                            \\-> vision model
                             \\
                              -> vision-language model (VLM) -> vision-language-action (VLA)

Every module is written to be read and run on a laptop (small models, CPU or a
single 8 GB GPU). Production systems use the same ideas at much larger scale via
libraries such as Hugging Face Transformers; the point here is to understand the
mechanisms so the larger systems make sense.

Modules
-------
- ``dl.attention``   : scaled dot-product attention and multi-head attention
- ``dl.transformer`` : positional encoding, transformer block, a small GPT
- ``dl.tokenizer``   : turning text into tokens (character level, plus BPE notes)
- ``dl.vision``      : a small CNN and a Vision Transformer (ViT)
- ``dl.multimodal``  : contrastive image-text alignment (CLIP) and cross-attention
                       fusion, the core mechanism inside a VLM and a VLA
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
