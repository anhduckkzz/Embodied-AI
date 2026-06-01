# Segment Anything: SAM, SAM 2, SAM 3

Three perspectives on the Segment Anything family, following the repo method:
mechanism first (from scratch), then framework (the real models), then real
system (how it is used in robotics and driving).

Code: from-scratch mechanism in [`dl/segmentation.py`](../../dl/segmentation.py),
runnable demo [`build_minisam.py`](build_minisam.py), framework usage in
[`sam_framework_guide.md`](sam_framework_guide.md), and the open-vocabulary
application in [`applications/open_vocab_segmentation/`](../../applications/open_vocab_segmentation/).

## 1. What problem SAM solves

Classic segmentation models are trained for a fixed set of classes. SAM
(Segment Anything Model, Meta, 2023) reframed segmentation as a promptable task:
given an image and a prompt (a point, a box, or a rough mask), output the mask of
the object the prompt indicates. Trained on SA-1B (over 1 billion masks), it
segments objects it was never explicitly labelled for, which is why it is called
a segmentation foundation model.

## 2. Mechanism: how promptable segmentation works (build it)

SAM has three parts, all implemented in miniature in
[`dl/segmentation.py`](../../dl/segmentation.py) as `MiniSAM`:

1. Image encoder. A heavy vision transformer runs once per image and produces a
   dense feature map. This is the expensive step, done a single time so that many
   prompts can be answered cheaply.
2. Prompt encoder. A point, box, or mask is turned into a few tokens (a point
   becomes a positional embedding; a learned output token represents the mask to
   be produced).
3. Mask decoder. A lightweight module fuses the prompt tokens with the image
   features using two-way attention (prompt tokens attend to the image, the image
   attends to the prompt), then turns the output token into a dynamic per-pixel
   classifier that is applied to the upscaled image features to produce the mask.

The defining property is promptability: the same image with a different prompt
produces a different mask. The runnable [`build_minisam.py`](build_minisam.py)
trains this on synthetic shapes and shows exactly that, segmenting whichever shape
the point lands on. SAM also predicts several candidate masks plus a confidence
score to handle ambiguity (a point on a shirt could mean the shirt or the person);
the mechanism is the same.

```bash
python curriculum/part6_perception/build_minisam.py
```

## 3. SAM 2: adding memory for video

SAM 2 (2024) generalizes SAM from images to video, so you can segment and track
an object across frames. It keeps the three parts above and adds a streaming
memory:

- Image encoder. Upgraded to a hierarchical encoder (Hiera) that yields
  multiscale features and processes frames one at a time (streaming).
- Memory attention. The features of the current frame are conditioned, by
  cross-attention, on memories of the target from past frames and on the original
  prompt. This is how the object is tracked through time.
- Memory encoder and memory bank. Store past predictions and features so future
  frames can attend to them.

The mental model: SAM segments a single image from a prompt; SAM 2 carries a
memory of the target so a prompt on frame 1 keeps segmenting that same object on
every later frame, even through occlusion. The new ingredient over the mechanism
in section 2 is the memory cross-attention.

## 4. SAM 3: promptable concept segmentation

SAM 3 (released November 19, 2025) adds the ability to segment by concept, not
just by a single geometric prompt. Its headline capability is Promptable Concept
Segmentation (PCS): given a short open-vocabulary text phrase (for example "yellow
school bus") or one or more image exemplars, find and segment all instances of
that concept in an image or video.

What is new compared to SAM and SAM 2:

- Text and exemplar prompts, not only points and boxes. This removes the
  fixed-label-set constraint and makes segmentation open-vocabulary. It is the
  point where SAM and CLIP-style language grounding (see the CLIP lesson) meet
  inside one model.
- Exhaustive instance segmentation of a concept (every matching object), rather
  than one object per prompt.
- A presence token in the architecture that helps the model discriminate between
  closely related phrases (for example "a player in white" versus other players).
- A large data engine that annotated over 4 million unique concepts, and a new
  benchmark, SA-Co, with about 270K concepts.

So the family is a clear progression: SAM segments one prompted object in an
image; SAM 2 tracks it through video with memory; SAM 3 segments all instances of
a language-or-exemplar concept in images and video.

## 5. Framework: running the real models

You do not train these from scratch. See
[`sam_framework_guide.md`](sam_framework_guide.md) for installing and calling the
released checkpoints (the official `segment-anything`, `sam2`, and `sam3`
packages, plus convenient wrappers in Ultralytics and Hugging Face), including
point, box, text, and exemplar prompts, and video tracking.

## 6. Real system: where SAM is used

- Robotics manipulation. Segment the object named in an instruction to get a
  precise mask, then estimate its pose or grasp (Part 9). SAM 3 text prompts make
  this language-conditioned directly; with SAM or SAM 2 you pair SAM with CLIP for
  the language part (the open-vocabulary application below).
- Autonomous driving. Instance masks for vehicles, pedestrians, and lanes;
  tracking with SAM 2 across frames; auto-labelling data engines (Part 11).
- Data annotation. SAM dramatically speeds up labelling segmentation datasets, a
  major practical use.
- Feeding a VLA. A concept mask is a strong grounding signal: it tells the policy
  exactly which pixels correspond to the referenced object (Part 10).

The application [`applications/open_vocab_segmentation/`](../../applications/open_vocab_segmentation/)
shows the classic combination, SAM for masks plus CLIP for labels, which gives
open-vocabulary segmentation even with the original SAM, and is conceptually what
SAM 3 unifies into a single model.

## Project

1. Run [`build_minisam.py`](build_minisam.py) and confirm the same scene yields
   different masks for different points. Increase the number of shapes and the
   image size and observe the effect on IoU.
2. Add a box prompt to `MiniSAM` (encode two corner points) and train with box
   prompts; compare to point prompts.
3. Install a real SAM 2 (framework guide) and segment-then-track an object in a
   short video clip.
4. Build the open-vocabulary application: SAM masks plus CLIP labels, to segment
   an object named by free text.

## Check your understanding

1. What are the three components of SAM, and which one runs only once per image?
2. What does it mean for segmentation to be promptable?
3. What does SAM 2 add over SAM, and how is the target tracked across frames?
4. What is promptable concept segmentation, and how is it different from a single
   point prompt?
5. How would you use SAM together with CLIP to segment an object named by text,
   and how does SAM 3 change that workflow?

## Go deeper

- Kirillov et al. (2023), Segment Anything (SAM); SA-1B dataset.
- Ravi et al. (2024), SAM 2: Segment Anything in Images and Videos; SA-V dataset.
- Meta AI (2025), SAM 3: Segment Anything with Concepts (arXiv:2511.16719);
  SA-Co benchmark; the Segment Anything Playground demo.
- Repositories: facebookresearch/segment-anything, facebookresearch/sam2,
  facebookresearch/sam3; Ultralytics SAM docs.
