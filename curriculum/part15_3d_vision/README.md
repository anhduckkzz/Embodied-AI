# Part 15 — 3D Vision: geometry, reconstruction, and neural scenes

> Deepens Part 6 with the **geometry of seeing in 3D** and the modern neural
> scene representations (NeRF, Gaussian Splatting, occupancy). This is how robots
> and cars build metric 3D understanding from cameras — and the substrate of
> spatial reasoning in VLAs.

## 1. Camera geometry (the foundation)

- **Pinhole + intrinsics `K`:** project a 3D camera-frame point to a pixel
  `[u v 1]ᵀ ∝ K [X Y Z]ᵀ` (Part 5). With the camera's pose (extrinsics, Part 3),
  this is the full **projection** `x = K [R|t] X`.
- **Lens distortion** + **calibration** recover `K` and distortion from a
  checkerboard (OpenCV `calibrateCamera`).
- **Homogeneous coordinates & projective geometry** — why a 4×4 / 3×4 matrix and a
  scale factor; points at infinity, lines, planes.

## 2. Multi-view geometry (3D from 2+ images)

- **Epipolar geometry:** two views of a point are linked by the **fundamental /
  essential matrix**; a point in one image lies on a line (epipolar) in the other.
  This constrains matching and recovers relative camera pose.
- **Triangulation:** given the same point in 2 known views, intersect the rays →
  its 3D position.
- **Stereo:** rectify two views, match → **disparity** → depth `Z = fB/d` (Part 5).
- **Structure from Motion (SfM):** from many unordered images, jointly recover
  camera poses **and** a sparse 3D point cloud (feature matching + **bundle
  adjustment**, the same nonlinear least-squares as graph SLAM, Part 7). Tools:
  COLMAP.
- **Visual odometry / VIO** — the online, sequential cousin (Part 7).

## 3. Dense depth & representations

- **Monocular depth** (learned): MiDaS, **Depth Anything** — per-pixel depth from
  one image (scale-ambiguous but powerful priors).
- **Multi-View Stereo (MVS):** dense reconstruction from many views.
- **3D representation zoo** (how you *store* 3D):
  - **Point clouds** (Part 6) — unordered XYZ; native LiDAR.
  - **Voxels / occupancy grids** — 3D pixels; "is this cell occupied?" (occupancy
    networks; **occupancy prediction** is now central in self-driving).
  - **Meshes** — vertices + faces; graphics-friendly.
  - **Signed Distance Fields (SDF)** — distance-to-surface function; smooth,
    great for optimization & reconstruction (DeepSDF).

## 4. Neural scene representations (the modern wave)

- **NeRF (Neural Radiance Fields):** a small MLP maps `(x, y, z, view-direction)`
  → color + density; **volume-render** rays to synthesize novel views. Learns a
  photorealistic 3D scene from posed images. Slow to train/render originally.
- **3D Gaussian Splatting (3DGS):** represent the scene as millions of 3D
  Gaussians, rasterized in **real time** with high quality — now the dominant
  approach for fast, high-fidelity reconstruction. Increasingly used for robot/AV
  scene reconstruction and simulation (Part 4 "digital twins").
- **Why robots care:** dense, differentiable 3D maps for navigation/manipulation,
  photoreal sim from real captures, and 6-DoF pose/grasp reasoning.

## 5. 3D deep learning (operating on 3D data)
- **PointNet / PointNet++** — networks that consume raw point clouds directly
  (permutation-invariant); the basis of much 3D detection/segmentation.
- **Sparse convolutions** (MinkowskiNet, SpConv) — efficient CNNs on sparse
  voxels; backbone of LiDAR detectors (Part 6).
- **BEV (Bird's-Eye-View)** — project features to a top-down grid for driving
  perception (BEVFormer, LSS) — fuses cameras (+LiDAR) into one planar map.
- **Transformers on 3D / multimodal** — increasingly unify it all (Part 16).

## 6. Where it lands
- **Perception (Part 6):** dense depth, 3D detection, occupancy.
- **SLAM (Part 7):** SfM/bundle adjustment, dense mapping, NeRF/3DGS maps.
- **Simulation (Part 4):** reconstruct real scenes → train in them (sim-to-real).
- **VLA (Part 16):** spatial grounding — many VLAs add depth/point/3D tokens for
  better manipulation; 3D reasoning is an open frontier.

## 🛠️ Project (laptop-friendly + A100 for training)
1. 💻 Calibrate your webcam with OpenCV; undistort images; estimate the essential
   matrix between two photos and **triangulate** a few points.
2. 💻 Run **COLMAP** SfM on ~30 phone photos of an object → camera poses + sparse
   cloud. Inspect in a viewer.
3. 🅰️ Train a **NeRF** or **3D Gaussian Splatting** scene (nerfstudio / gsplat) on
   that capture; render novel views. Relate volume rendering to the geometry above.
4. 💻 Run a pretrained **Depth Anything** on images; back-project to a point cloud
   using `K` (ties to Part 6's pipeline).

## ✅ Check your understanding
1. Write the full projection from a world point to a pixel (intrinsics+extrinsics).
2. What does epipolar geometry constrain, and how does triangulation use it?
3. How is SfM's bundle adjustment related to graph SLAM (Part 7)?
4. Contrast point clouds, voxels/occupancy, meshes, and SDFs.
5. NeRF vs 3D Gaussian Splatting — what changed to make it real-time?

## 📖 Go deeper
- Hartley & Zisserman, *Multiple View Geometry* (the bible). Szeliski, *Computer
  Vision*. COLMAP & nerfstudio docs.
- Papers: NeRF (Mildenhall 2020), **3D Gaussian Splatting** (Kerbl 2023),
  PointNet (Qi 2017), DeepSDF (Park 2019), Depth Anything (2024).

➡️ **Next:** [Part 16 — Foundation Models](../part16_foundation_models/): the
transformers and diffusion models that power VLAs.
