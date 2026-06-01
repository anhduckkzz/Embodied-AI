# Framework guide: running real SAM, SAM 2, and SAM 3

The from-scratch [`dl/segmentation.py`](../../dl/segmentation.py) teaches the
mechanism. For real work you use the released models. This guide shows how to
install and call them. None of this runs in the curriculum sandbox (the models
need downloaded checkpoints and a GPU), so treat the snippets as reference for
your own machine. On the RTX 4060 (8 GB) the smaller variants run for inference;
use the A100 for the large encoders or heavy video.

## Which package

- Official Meta repositories: `facebookresearch/segment-anything` (SAM),
  `facebookresearch/sam2` (SAM 2), `facebookresearch/sam3` (SAM 3). These provide
  inference code, checkpoints, and example notebooks.
- Ultralytics wraps SAM, SAM 2, and SAM 3 behind one simple API.
- Hugging Face `transformers` provides SAM and SAM 2 model classes and the
  AutoModel interface for easy loading.

Pick the official repo to follow the paper closely, Ultralytics for the quickest
start, or Hugging Face if you already use that stack.

## SAM: prompt a single image

```python
# pip install segment-anything  (and download a checkpoint)
from segment_anything import sam_model_registry, SamPredictor
import cv2

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth").to("cuda")
predictor = SamPredictor(sam)

image = cv2.cvtColor(cv2.imread("scene.jpg"), cv2.COLOR_BGR2RGB)
predictor.set_image(image)                       # runs the heavy encoder once
masks, scores, _ = predictor.predict(            # cheap, repeat for many prompts
    point_coords=[[420, 300]], point_labels=[1])  # one positive click
```

`set_image` runs the expensive image encoder once; each `predict` with a new
point or box is cheap. That split is exactly the mechanism from the lesson.

## SAM 2: segment and track through a video

```python
# pip install sam2  (from the official repo; download a checkpoint)
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor("sam2_hiera_b.yaml", "sam2_hiera_b.pt")
state = predictor.init_state(video_path="clip.mp4")
# Prompt the object on the first frame with a click:
predictor.add_new_points(state, frame_idx=0, obj_id=1,
                         points=[[400, 250]], labels=[1])
# Propagate the mask through the whole video using the memory:
for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
    ...   # mask_logits holds the tracked object's mask for this frame
```

The memory carries the target from frame 0 to every later frame; you prompt once
and the object is tracked.

## SAM 3: segment by concept (text or exemplar)

```python
# pip install sam3  (official repo; download a checkpoint)
# Conceptual usage; check the repo for the exact current API.
from sam3 import SAM3

model = SAM3.from_pretrained("sam3_base").to("cuda")
# Promptable Concept Segmentation: a short open-vocabulary noun phrase finds and
# segments ALL matching instances in the image (or video).
result = model.predict(image, text_prompt="yellow school bus")
masks = result.masks          # one mask per detected instance of the concept
```

SAM 3 also accepts image exemplars (give it one example crop, it finds all
similar instances) and still supports the point and box prompts of SAM and SAM 2.

## Quick start with Ultralytics (one API for all three)

```python
# pip install ultralytics
from ultralytics import SAM
model = SAM("sam2_b.pt")              # or a SAM / SAM 3 checkpoint name
results = model("scene.jpg", points=[[420, 300]], labels=[1])
```

## Practical notes

- Run the image encoder once, then issue many prompts; do not re-encode per
  prompt.
- Checkpoints come in sizes (for example vit_b, vit_l, vit_h). Smaller fits the
  4060 and runs faster; larger is more accurate.
- For real-time or long video, prefer the smaller SAM 2 or SAM 3 variants and
  consider the faster real-time releases.
- To label an object named by free text with SAM or SAM 2 (no native text
  prompt), pair them with CLIP, see the open-vocabulary application. With SAM 3
  the text prompt is built in.

## Where to go next

- Lesson and mechanism: [`segment_anything.md`](segment_anything.md).
- Application combining SAM and CLIP:
  [`applications/open_vocab_segmentation/`](../../applications/open_vocab_segmentation/).
