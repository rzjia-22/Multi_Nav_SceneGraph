# NavDiffusion V0

NavDiffusion V0 is the first project-owned model trained from the frozen
Dataset V0. Its purpose is deployment-oriented goal navigation in the accepted
Isaac Research Forest and, after later closed-loop validation, on a standing
DIABLO with a RealSense D435i. This milestone covers data windows, training,
checkpointing and ROS loading. It deliberately does not report test-split or
closed-loop results.

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
not evidence of closed-loop success. The test split was not loaded for
statistics, training, selection or evaluation.

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
checkpoint, five-frame history and eight-waypoint control output successfully.
Legacy ForestNavigation checkpoint loading remains available as
`model_backend=legacy`.

```bash
make navdiffusion-v0-data
make navdiffusion-v0-single-batch
make navdiffusion-v0-overfit
make navdiffusion-v0-train
make navdiffusion-v0-smoke
make navdiffusion-v0-ros-smoke

MNS_DIFFUSION_BACKEND=mns_v0 \
MNS_DIFFUSION_CHECKPOINT=/workspace/models/trained/navdiffusion_v0/best.pt \
make phase1-diffusion
```

The final command is the next-stage closed-loop entry point; it was not run as
part of this training milestone. Test evaluation, Isaac closed-loop acceptance
and real DIABLO/D435i trials remain intentionally deferred.
