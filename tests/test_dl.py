"""Tests for the dl.* architecture library: attention, transformer/GPT, vision,
multimodal. These check both correct shapes and that the components actually
learn (a tiny model should overfit a tiny task)."""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn.functional as F  # noqa: E402


# ----------------------------------------------------------- attention ----
def test_attention_shapes_and_weights_sum_to_one():
    from dl.attention import scaled_dot_product_attention
    B, H, T, D = 2, 3, 5, 8
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, T, D)
    assert w.shape == (B, H, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(B, H, T), atol=1e-5)


def test_causal_mask_blocks_future():
    from dl.attention import causal_mask, scaled_dot_product_attention
    T, D = 4, 6
    q, k, v = (torch.randn(1, 1, T, D) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v, mask=causal_mask(T))
    # Upper triangle (future) must have ~0 weight.
    upper = torch.triu(w[0, 0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


def test_multihead_self_and_cross_attention_shapes():
    from dl.attention import MultiHeadAttention
    mha = MultiHeadAttention(dim=16, num_heads=4)
    x = torch.randn(2, 7, 16)
    assert mha(x).shape == (2, 7, 16)               # self-attention
    ctx = torch.randn(2, 5, 16)
    assert mha(x, context=ctx).shape == (2, 7, 16)  # cross-attention


# --------------------------------------------------------- transformer ----
def test_gpt_forward_and_generate_shapes():
    from dl.transformer import GPT
    model = GPT(vocab_size=20, dim=32, depth=2, num_heads=4, max_len=16)
    idx = torch.randint(0, 20, (2, 8))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 8, 20)
    assert loss.item() > 0
    out = model.generate(idx[:, :1], max_new_tokens=5)
    assert out.shape == (2, 6)


def test_gpt_overfits_a_short_sequence():
    """A small GPT must be able to memorize one short string (loss -> near 0)."""
    from dl.tokenizer import CharTokenizer
    from dl.transformer import GPT
    torch.manual_seed(0)
    text = "hello transformers, attention is the core mechanism."
    tok = CharTokenizer(text)
    data = tok.encode_tensor(text)
    x = data[:-1].unsqueeze(0)
    y = data[1:].unsqueeze(0)
    model = GPT(tok.vocab_size, dim=64, depth=2, num_heads=4, max_len=len(text))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for step in range(300):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.2   # clearly learned the sequence


# -------------------------------------------------------------- vision ----
def test_cnn_and_vit_forward_shapes():
    from dl.vision import SimpleCNN, ViT
    imgs = torch.randn(4, 3, 32, 32)
    assert SimpleCNN(num_classes=10)(imgs).shape == (4, 10)
    vit = ViT(image_size=32, patch_size=8, num_classes=10, dim=64, depth=2, num_heads=4)
    assert vit(imgs).shape == (4, 10)
    assert vit(imgs, return_features=True).shape == (4, 64)


def test_vit_can_learn_a_trivial_pattern():
    """ViT should separate all-bright vs all-dark images quickly."""
    from dl.vision import ViT
    torch.manual_seed(0)
    vit = ViT(image_size=16, patch_size=4, num_classes=2, dim=64, depth=2, num_heads=4)
    opt = torch.optim.Adam(vit.parameters(), lr=1e-3)
    for _ in range(60):
        bright = torch.rand(8, 3, 16, 16) * 0.2 + 0.8
        dark = torch.rand(8, 3, 16, 16) * 0.2
        x = torch.cat([bright, dark])
        y = torch.cat([torch.ones(8), torch.zeros(8)]).long()
        logits = vit(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert (vit(x).argmax(-1) == y).float().mean() > 0.9


# ---------------------------------------------------------- multimodal ----
def test_clip_alignment_improves():
    """Training a dual encoder should make matched image-text pairs most similar."""
    from dl.multimodal import DualEncoder, clip_contrastive_loss
    torch.manual_seed(0)
    n, idim, tdim = 16, 32, 24
    # Create paired features: text feature is a fixed linear function of image
    # feature plus noise, so a real correspondence exists to be discovered.
    img_feats = torch.randn(n, idim)
    W = torch.randn(idim, tdim)
    txt_feats = img_feats @ W + 0.1 * torch.randn(n, tdim)

    model = DualEncoder(idim, tdim, embed_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        ie, te = model(img_feats, txt_feats)
        loss = clip_contrastive_loss(ie, te)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first    # learned to align
    # The diagonal (true pairs) should dominate each row of the similarity matrix.
    ie, te = model(img_feats, txt_feats)
    sim = F.normalize(ie, dim=-1) @ F.normalize(te, dim=-1).t()
    assert (sim.argmax(dim=1) == torch.arange(n)).float().mean() > 0.7


def test_cross_attention_fusion_shape():
    from dl.multimodal import CrossAttentionFusion
    fusion = CrossAttentionFusion(dim=32, num_heads=4)
    text_tokens = torch.randn(2, 6, 32)     # queries
    image_tokens = torch.randn(2, 10, 32)   # keys/values
    out = fusion(text_tokens, image_tokens)
    assert out.shape == (2, 6, 32)
