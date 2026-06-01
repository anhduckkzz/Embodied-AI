"""Tests for the from-scratch CLIP and SAM mechanisms in dl/.

These confirm the mechanisms work: MiniCLIP learns to align images and text well
enough to classify zero-shot, and MiniSAM learns promptable segmentation
(a point inside a shape yields that shape's mask)."""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn.functional as F  # noqa: E402


# ---------------------------------------------------------------- CLIP ----
def test_miniclip_zero_shot_classification():
    from dl.clip import (CLIP_MAX_LEN, CLIP_VOCAB, MiniCLIP, class_token_ids,
                        synthetic_color_dataset)
    torch.manual_seed(0)
    images, tokens, labels = synthetic_color_dataset(1024, image_size=16, seed=0)
    model = MiniCLIP(image_size=16, patch_size=4, vocab_size=len(CLIP_VOCAB),
                     max_text_len=CLIP_MAX_LEN, dim=96, embed_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    n = len(images)
    for _ in range(250):
        idx = torch.randint(0, n, (64,))
        loss = model.contrastive_loss(images[idx], tokens[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    # Zero-shot: classify a fresh set by comparing to the class-caption texts.
    test_imgs, _, test_labels = synthetic_color_dataset(200, image_size=16, seed=99)
    logits = model.zero_shot_logits(test_imgs, class_token_ids())
    acc = (logits.argmax(-1) == test_labels).float().mean().item()
    assert acc > 0.8, f"zero-shot accuracy too low: {acc}"


# ----------------------------------------------------------------- SAM ----
def test_minisam_forward_shape():
    from dl.segmentation import MiniSAM
    model = MiniSAM(image_size=32, dim=64)
    image = torch.randn(4, 3, 32, 32)
    points = torch.rand(4, 2)
    assert model(image, points).shape == (4, 32, 32)


def test_minisam_learns_promptable_segmentation():
    from dl.segmentation import MiniSAM, dice_bce_loss, mask_iou, synthetic_shapes
    torch.manual_seed(0)
    images, points, masks = synthetic_shapes(1024, image_size=32, seed=0)
    model = MiniSAM(image_size=32, dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(images)
    for _ in range(600):
        idx = torch.randint(0, n, (32,))
        logits = model(images[idx], points[idx])
        loss = dice_bce_loss(logits, masks[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    test_i, test_p, test_m = synthetic_shapes(128, image_size=32, seed=7)
    pred = model.predict(test_i, test_p)
    iou = mask_iou(pred, test_m)
    assert iou > 0.55, f"segmentation IoU too low: {iou}"
