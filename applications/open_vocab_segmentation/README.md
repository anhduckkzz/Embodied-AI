# Open-vocabulary segmentation (application)

Segment objects named by free text, for example "the red mug" or "every
pedestrian". This is an application-layer component built on the real frameworks
(SAM and CLIP, or SAM 3). It deliberately does not import the from-scratch
teaching code in [`dl/`](../../dl/); learn the mechanism there, build the system
here.

Code: [`pipeline.py`](pipeline.py). Mechanism and background: the
[SAM lesson](../../curriculum/part6_perception/segment_anything.md) and the
[CLIP lesson](../../curriculum/part16_foundation_models/04_vlm_multimodal/clip.md).

## The idea

Neither classic model alone is both localized and language-aware:

- CLIP understands text but classifies a whole image; it does not produce masks.
- SAM (and SAM 2) produce precise masks but, before SAM 3, take only point, box,
  or mask prompts, not text.

Combine them and you get open-vocabulary segmentation:

```
image -> SAM (propose masks for everything) -> for each mask:
            crop the region -> CLIP embed -> compare to the text queries
         -> keep masks whose best text score passes a threshold
```

SAM 3 unifies this into one model: give it a text phrase and it returns all
matching instances directly, so the CLIP step is unnecessary. The pipeline
supports both paths and prefers SAM 3 when available.

## Run it

The models need downloaded checkpoints and a GPU, so this runs on your machine,
not in the curriculum sandbox. The script explains what to install if the models
are absent.

```bash
python -m applications.open_vocab_segmentation.pipeline
```

```python
import cv2
from applications.open_vocab_segmentation.pipeline import open_vocabulary_segment

image = cv2.cvtColor(cv2.imread("scene.jpg"), cv2.COLOR_BGR2RGB)
results = open_vocabulary_segment(image, ["mug", "keyboard", "person"])
# results: list of (query, score, mask) for SAM+CLIP, or (query, mask) for SAM 3.
```

## Why this matters for the curriculum's applications

- Robotics manipulation (Part 9, Part 10). Turn an instruction such as "pick up
  the red mug" into a precise mask of the referenced object, then estimate its
  pose or grasp. The mask is a strong grounding signal for a VLA.
- Autonomous driving (Part 11). Open-vocabulary instance masks for vehicles,
  pedestrians, and unusual objects, including the long-tail classes a fixed-label
  detector misses; track them across frames with SAM 2.
- Data engines. Auto-label segmentation datasets by text, a major practical use
  that powers the data flywheel behind these models.

## On your hardware

- RTX 4060 (8 GB): run the smaller SAM and a base CLIP for inference. Keep image
  resolution modest and process a few prompts at a time.
- A100: larger encoders, video tracking with SAM 2, and SAM 3 concept
  segmentation at scale.

## Extend it

1. Add non-maximum suppression across masks so overlapping detections of the same
   object are merged.
2. Feed the resulting mask into the VLA in [`applications/vla/`](../vla/) as an
   extra input channel, so the policy is told exactly which pixels are the target.
3. Swap SAM 2 in for video and track the named object across a clip.
