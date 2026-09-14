# NavDiffusion V0

NavDiffusion V0 is the first project-owned model trained from the frozen
Dataset V0. Its purpose is deployment-oriented goal navigation in the accepted
Isaac Research Forest and, after staged validation, on a standing DIABLO with
a RealSense D435i. Training is complete. The matching-domain Isaac deployment
gate, a subsequent exhaustive ten-mission validation analysis, and the final
one-pass held-out test are recorded below. Validation reached 4/10 and test
reached 3/10; each recorded six conservative collisions. These results do not
authorize real inference or motion deployment. Dataset V0 test has now been
opened for this final V0 evaluation.

## Data authority and cache

`datasets/dataset_v0/dataset_index.json` and its HDF5 v2 episodes remain the
only source of truth. The training loader selects exactly 50 `train` episodes
and 10 `validation` episodes from the index and rejects `test`. It creates a
disposable 1,540,488,499-byte cache under `runs/cache/navdiffusion_v0`; the
cache is invalidated by the dataset-index SHA256, preprocessing-config SHA256,
or any selected HDF5 SHA256. It is ignored by Git and may be deleted at any
time.

The frozen window audit produced 5,357 train and 1,081 validation windows.
Four initial anchors per episode are dropped rather than history-padded. At
each remaining 10 Hz anchor, 32 future positions at 0.1 s spacing are
interpolated from the 50 Hz executed robot state and transformed into the
anchor robot planar frame. Episode tails repeat the final executed pose:
29.923% of train windows contain at least one padded point, the mean is 4.937
padded points per window, and 0.952% are fully padded.

## Shared preprocessing

`mns_navigation.navdiffusion_v0.NavDiffusionPreprocessor` is used by both the
window dataset and `MNSNavDiffusionPredictor`:

- RGB is resized from 640×360 to 160×90 with bilinear interpolation, scaled by
  255, and standardized with ImageNet mean `[0.485, 0.456, 0.406]` and standard
  deviation `[0.229, 0.224, 0.225]`.
- Raw Z16 depth is decoded with the per-episode scale and passed through the
  existing `research_data.depth` z-buffer registration into the RGB optical
  frame. Invalid/non-positive values and registration holes become 10 m;
  metric depth is clipped to 10 m, divided by 10, resized with nearest-neighbor,
  then standardized with train-only mean `0.5860419118` and standard deviation
  `0.3728206109`. The post-resize invalid/far fraction is 39.335%.
- Five real frames form the history; no first-frame replication is used.
- The mission goal is transformed to the same anchor body frame and divided by
  10 m.
- Executed future XY is normalized by the fixed symmetric ±2.5 m range. The
  observed train maximum is 2.08 m, so no label is clipped.

No RGB, depth, crop, flip or other augmentation is active in V0. The exact
resolved contract and hashes are stored in `models/trained/navdiffusion_v0/`
and embedded in the checkpoint.

## Network and initialization

The architecture follows the audited ForestNavigation NavDiffusion design at
revision `0b29c399754f510499bfe9cc9d592cba215a7161`, but is a project-owned
plain-PyTorch implementation:

```text
5 × 4 × 90 × 160 RGB-D
  -> torchvision EfficientNet-B0 ImageNet encoder
  -> 256-d frame embeddings + 256-d goal token
  -> 4-layer, 4-head Transformer (FF factor 4)
  -> 256-d condition
  -> Conditional 1-D U-Net [64, 128, 256]
  -> 10-step squared-cosine DDPM, epsilon prediction
  -> 32 × 2 local trajectory
```

The torchvision 0.22.1 `IMAGENET1K_V1` EfficientNet weights came from the
official PyTorch URL and have SHA256
`7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`.
The original RGB stem channels are copied exactly into the four-channel stem;
the depth channel starts at exactly zero. All 49 native EfficientNet BatchNorm
layers and pretrained state are retained. The network has 12,675,270 total and
trainable parameters.

## Training result

Training used PyTorch 2.7.1/CUDA 12.8 on the RTX 4060 Laptop GPU. AdamW used
learning rates `1e-4` for EfficientNet and `3e-4` for the goal encoder,
Transformer and diffusion U-Net, weight decay `1e-4`, gradient clipping 5,
five warmup epochs and cosine decay. The actual micro-batch was 16 with two
gradient-accumulation steps (effective 32) under bf16 AMP.

The single-batch gate passed with finite loss, nonzero gradients and 1,140 MiB
peak allocated VRAM. The fixed 32-window overfit gate reduced loss from
1.05190 to 0.01754 (98.33%) and ADE from 2.681 m to 0.282 m; checkpoint reload
was numerically identical. Full training stopped at epoch 120 after 20 epochs
without a better fixed-seed validation ADE. It took 4,315.1 s and peaked at
1,261.8 MiB allocated VRAM. The selected epoch 100 checkpoint achieved:

- validation ADE: 0.090123 m;
- validation FDE: 0.169597 m;
- best observed validation diffusion loss: 0.006512.

These are open-loop metrics on ten validation episodes from two unseen scenes,
not evidence of closed-loop success. During data statistics, training,
early-stopping and checkpoint selection, the test split was not loaded. It was
first opened only after the frozen V0 system and validation analysis were
complete, for the final closed-loop evaluation documented below.

## Artifacts and runtime

The formal directory is `models/trained/navdiffusion_v0/`. `best.pt` is a
142,598,595-byte Git LFS object with SHA256
`7b3bca67af26c2d839555e9b27a7a420762b572fa88664a227986174d9a68ae0` and
checkpoint format version 1. It contains model, preprocessing and architecture
metadata plus optimizer/scheduler/scaler state for traceability. `last.pt` and
derived caches remain ignored runtime artifacts.

The final predictor smoke test reproduced the cached training input exactly,
loaded on CPU and CUDA, and returned a finite `[32,2]` trajectory. Warm
10-step sampling measured 60.7 ms on CUDA and 127.5 ms on CPU in the recorded
run. The real `DiffusionNavigatorNode` loaded `model_backend=mns_v0`, the
checkpoint and five-frame history successfully. It now retains and publishes
all 32 points from one sample for diagnostics while the unchanged controller
uses the first eight. Legacy ForestNavigation checkpoint loading remains
available as `model_backend=legacy`.

## First matching-domain closed loop

The dedicated `/diablo_1` runtime uses the same `build_research_forest(...)`,
actual `TerrainSurfaceQuery`, kinematic DIABLO standing profile and two-camera
D435i simulation as Dataset V0. Raw 848×480 depth is registered into the
640×360 RGB frame by the shared `research_data.depth` implementation before
ROS publication. Isaac publishes only standard ROS 2 RGB, depth, CameraInfo,
odom, TF and clock messages. Navigation, Safety and arbitration remain in the
robotics-ML container; mapping/Hydra is disabled.

Gate A (`validation_scene_000_episode_000`) passed: 4.922 m executed, 0.348 m
goal error, 0.187 m minimum conservative clearance and no collision. Planning
ran at 1.87 Hz with 151.2 ms p95 inference. The arbiter selected the Safety
source for 0.18 s (1.60%). A short representative Gate B mission
(`validation_scene_001_episode_001`) also passed with 0.350 m goal error,
0.214 m clearance and no Safety override.

Gate B then failed fast on the long mission
`validation_scene_001_episode_004`: the surrogate intersected the conservative
proxy of `tree_021` at 9.74 s, after 5.179 m of motion, while still 3.638 m from
the goal. The full 32-point prediction first intersected a tree proxy at
6.97 s; at 9.47 s the eight control points also intersected it. Safety first
intervened at 9.66 s for 0.10 s, too late to avoid the collision. The final
RGB/depth frame visibly contains the near-field trunk and the recorded path
ends at that proxy. This is classified `MODEL`, rather than sensor,
controller, Safety or simulator failure: frames were fresh, trajectories were
finite and continuous, control and diagnostic paths came from the same sample,
and the control trajectory itself became unsafe before impact.

Gate B stopped at 1/3 as required. This historical fail-fast gate remains
separate from the exhaustive analysis mode.

The exhaustive analysis independently reran all 10 validation episodes in two
scene-batched Isaac lifecycles and continued after each mission failure. Four
missions reached the 0.35 m goal tolerance; six ended in conservative
tree-proxy collisions, with no timeout or stall. Success by route bucket was
short 2/3, medium 1/4 and long 1/3. The flat, medium-density
`validation_scene_000` reached 3/5, while the moderate, high-density
`validation_scene_001` reached 1/5. This ten-sample result suggests only a
modest route-length association; failures were more concentrated in the
second scene and on routes with slightly lower expert planning clearance.

Across 186 planning cycles, successful episodes had 6/114 unsafe full-horizon
predictions but 0/114 unsafe eight-point control predictions. Failed episodes
had 36/72 unsafe full predictions and 11/72 unsafe control predictions. Safety
intervened in four failed episodes for 1.74 s total and in no successful
episode; it did not prevent any of the six collisions. The prior
`validation_scene_001_episode_004` collision repeated on `tree_021` at 9.89 s
versus 9.74 s in Gate B, with 0.106 m normalized mean trajectory separation.
These are descriptive associations, not causal findings.

The run accumulated 100.64 s of simulated motion over 437.25 s complete wall
time. Inference across all planning cycles averaged 117.5 ms, with 262.1 ms
p95 and one 1189.6 ms cold/warm-up maximum. Every episode reported fresh RGB,
nonnegative history-ready timing, the frozen checkpoint hash, and zero
discarded cross-episode planning diagnostics after reset isolation.

The resulting readiness classification is **`NOT_READY`**. Sensor-only data
capture is still technically low risk, but this model is not ready for real
inference acceptance or motion. Machine-readable results, failure plots and
the ten-row analysis are under `artifacts/navdiffusion_v0_closed_loop/`;
high-volume traces remain ignored under `runs/navdiffusion_v0_closed_loop/`.

## Final held-out closed-loop test

The canonical one-pass test used the identical frozen checkpoint,
preprocessing, 2 Hz planner, 32-point prediction, first-eight-point Pure
Pursuit control, Safety settings, DIABLO surrogate, D435i cameras and Research
Forest builder. Hydra remained disabled. All ten missions from the two test
scenes were executed once in two scene-batched Isaac lifecycles; no train or
validation mission was executed by the test command, and no mid-test tuning or
per-episode rerun occurred.

Three missions reached the 0.35 m goal tolerance. Six missions ended in a
conservative tree-proxy collision, one mission followed the model command out
of the valid terrain domain, and none timed out or stalled. Bucket success was
short 1/3, medium 1/4 and long 1/3. The gentle, low-density, grass,
bright-lighting `test_scene_000` reached 3/5; the moderate, high-density,
bare-soil, normal-lighting `test_scene_001` reached 0/5 and accounted for five
collisions. Because these attributes co-vary by scene, this is an association,
not a causal attribution.

Validation and test are broadly similar: 4/10 versus 3/10 success, with six
collisions in each. Every test collision was preceded first by an unsafe
32-point prediction and then by an unsafe first-eight control prediction.
Across all test planning cycles, successful missions had 2.02% unsafe full
predictions and 0% unsafe control predictions; failed missions had 33.70% and
11.60% respectively. Safety intervened in 5/7 failures and 2/3 successes, but
did not avert a collision. Test failures were not monotonically associated
with route length: every bucket had exactly one success, and mean planned
length was 6.547 m for failures versus 6.404 m for successes. They were also
not concentrated at the most extreme camera perturbations.

The test accumulated 146.00 s of simulated motion over 583.52 s wall time.
Inference averaged 136.7 ms with 303.3 ms p95; a single 3366.9 ms maximum is
retained rather than hidden. The final V0 classification remains
**`NOT_READY`**. The reports and failure overlays are under
`artifacts/navdiffusion_v0_test/`, while high-volume traces remain ignored
under `runs/navdiffusion_v0_test/`.

Dataset V0 test was first opened for this NavDiffusion V0 final closed-loop
evaluation. Any NavDiffusion V1 design informed by these results must use new
held-out scenes for its own final evaluation; Dataset V0 test can no longer be
described as untouched for V1.

```bash
make navdiffusion-v0-data
make navdiffusion-v0-single-batch
make navdiffusion-v0-overfit
make navdiffusion-v0-train
make navdiffusion-v0-smoke
make navdiffusion-v0-ros-smoke
make navdiffusion-v0-closed-loop-gate
make navdiffusion-v0-closed-loop-analysis
make navdiffusion-v0-closed-loop-test

MNS_DIFFUSION_BACKEND=mns_v0 \
MNS_DIFFUSION_CHECKPOINT=/workspace/models/trained/navdiffusion_v0/best.pt \
make phase1-diffusion
```

The final `phase1-diffusion` command is the older Go2 integration-regression
path, not the DIABLO Research Forest acceptance path. Real DIABLO/D435i
inference or motion remains intentionally deferred. Hydra was not run during
either closed-loop evaluation.
