# Part 6 — Perception: from pixels & points to understanding

> Code: [`robotics/pointcloud.py`](../../robotics/pointcloud.py), demo:
> [`perception_demo.py`](perception_demo.py). Perception turns raw sensor data
> (Part 5) into the **objects, geometry, and semantics** that planning (Part 8)
> and VLA models (Part 10) act on. This is where deep learning meets robotics.

Two complementary questions perception answers:
- **"What is where?"** — detect/classify/segment objects (semantics).
- **"What's the geometry?"** — depth, surfaces, free space (3D structure).

---

## 1. 2D image perception (the deep-learning core)

Tasks, in increasing detail:
- **Classification** — one label per image ("cat").
- **Object detection** — boxes + labels. Two lineages: **YOLO** (one-stage,
  real-time — the autonomy default) and **Faster R-CNN** (two-stage, accurate).
  Output: boxes `(x, y, w, h)`, class, confidence; cleaned with **non-max
  suppression**.
- **Semantic segmentation** — class per *pixel* (road/car/person). Drivable-area
  for cars.
- **Instance / panoptic segmentation** — per-pixel *and* per-object (each car
  separately). Mask R-CNN, **SAM**.
- **Tracking** — link detections across frames (who's who over time) → velocities.

> The backbone has shifted from CNNs (ResNet) to **Vision Transformers (ViT)**
> and foundation encoders (**DINOv2, SigLIP, CLIP**). These same encoders are the
> "eyes" of VLA models in Part 10 — perception and VLA share machinery.

🅰️ Train detectors/segmenters on the A100; run inference (YOLO) in real time on
the 4060.

## 2. Depth & 3D from images

- **Monocular depth estimation** — predict per-pixel depth from one image
  (MiDaS, Depth Anything). Learned, scale-ambiguous but improving fast.
- **Stereo matching** — disparity → metric depth (Part 5).
- **Structure from Motion (SfM) / Multi-View Geometry** — recover 3D + camera
  poses from many images (the classical backbone of mapping; see also NeRF /
  3D Gaussian Splatting for neural scene reconstruction).

## 3. Point-cloud processing (LiDAR / depth) — hands-on here

A point cloud is an `(N,3)` array. The classic pipeline — all implemented in
[`robotics/pointcloud.py`](../../robotics/pointcloud.py) and runnable in the demo:

1. **Filter / crop** — `passthrough_filter` to keep a region of interest.
2. **Downsample** — `voxel_downsample` to a regular grid (millions → thousands).
3. **Ground/plane segmentation** — `fit_plane_ransac` robustly finds the road/
   floor so you can separate it from objects above it.
4. **Clustering** — group the remaining points into object instances
   (Euclidean/DBSCAN clustering).
5. **Registration** — `icp` aligns two clouds (scan-to-scan = LiDAR odometry,
   scan-to-map = localization). This feeds SLAM (Part 7).

```python
from robotics.pointcloud import voxel_downsample, fit_plane_ransac, icp
cloud = voxel_downsample(raw_lidar, 0.1)        # thin it out
plane, ground_mask = fit_plane_ransac(cloud)    # find the road
objects = cloud[~ground_mask]                   # everything standing on it
```

For production use **Open3D** or **PCL** (KD-trees, fast clustering, normals);
the code here shows you what they do underneath.

## 4. 3D object detection (the AV workhorse)

Detect **oriented 3D boxes** (position, size, heading, class) — what a car needs
to know "there's a vehicle 12 m ahead, moving left." Input modalities:
- **LiDAR-based:** **PointPillars**, **CenterPoint**, **VoxelNet** — voxelize/
  pillarize the cloud, run a CNN/transformer, regress 3D boxes. Accurate geometry.
- **Camera-based (BEV):** lift image features into a **Bird's-Eye-View** grid
  (**BEVFormer**, **LSS**) — cheaper sensors, harder depth.
- **Fusion:** combine LiDAR geometry + camera semantics for the best of both.

Datasets/benchmarks: **KITTI**, **nuScenes**, **Waymo Open**. 🅰️ training is
A100 territory; pretrained inference can run on the laptop.

## 5. Sensor fusion — one coherent world

No single sensor suffices, so fuse:
- **Early fusion** — combine raw data (paint LiDAR points with camera color).
- **Late fusion** — run detectors per sensor, merge the outputs.
- **Mid / deep fusion** — fuse learned features (most modern nets, e.g. BEV
  fusion).
- **Geometric prerequisite:** everything must be in a common frame via accurate
  **extrinsic calibration** + **transforms** (Part 3). A 1° error smears the
  fused result.

Temporal fusion (tracking + filtering, Part 7) adds the time dimension:
detections become tracked objects with velocity and predicted motion.

## 6. How perception feeds the rest of the stack
- → **SLAM (7):** features/point clouds for localization & mapping.
- → **Planning (8):** an occupancy/cost map of free vs blocked space + obstacles.
- → **VLA (10):** image (and sometimes depth/point) tokens conditioning the policy.
- → **Decision (8/11):** tracked objects + predictions for "what will that
  pedestrian do?"

## 🛠️ Project
Run [`perception_demo.py`](perception_demo.py): it simulates a 2D LiDAR in a
scene, builds a point cloud, RANSAC-segments the ground from a synthetic 3D
cloud, voxel-downsamples, and ICP-aligns two scans — the full classical pipeline,
no GPU needed. Then swap in **Open3D** and a real `.pcd`/KITTI scan.

## ✅ Check your understanding
1. Detection vs semantic vs instance segmentation — what does each output?
2. Why voxel-downsample before processing a LiDAR cloud?
3. What does RANSAC give you on a road scene, and why is it robust?
4. LiDAR-based vs camera-BEV 3D detection — trade-offs?
5. Early vs late vs deep sensor fusion — define each.

## 📖 Go deeper
- Szeliski, *Computer Vision* (detection, segmentation, multi-view).
- Open3D & PCL tutorials (point clouds). · KITTI / nuScenes dev-kits.
- Papers: YOLO, Mask R-CNN, SAM, PointPillars, CenterPoint, BEVFormer, Depth Anything.

➡️ **Next:** [Part 7 — State Estimation & SLAM](../part7_state_estimation_slam/):
fuse motion + perception over time to know *where you are* and *build a map*.
