# NavDiffusion V0 frozen baseline

NavDiffusion V0 is the first project-owned visual-navigation baseline trained
from Dataset V0. Training and its final validation/test lifecycle are complete.
The model is retained for reproducibility and comparison; it is not the active
deployment candidate. Final readiness is **`NOT_READY`**.

The current-tree evidence authority is:

- `artifacts/baselines/navdiffusion_v0/results.json` — machine-readable facts;
- `artifacts/baselines/navdiffusion_v0/summary.md` — concise interpretation;
- one representative validation failure and one representative test failure;
- `models/trained/navdiffusion_v0/best.pt` — frozen Git LFS checkpoint.

Detailed Gate A/B, exhaustive-validation and one-pass-test artifacts are
recoverable from annotated tag `pre-cleanup-navdiffusion-v0-final`. They are no
longer active workflow inputs or repository tests.

## Data and preprocessing contract

`datasets/dataset_v0/dataset_index.json` and the HDF5 v2 episodes are the only
raw-data source of truth. Training uses 50 train episodes and model selection
uses 10 validation episodes. It produced 5,357 train and 1,081 validation
windows. The test split was not used for preprocessing statistics, training,
early stopping or checkpoint selection.

Training and runtime share
`mns_navigation.navdiffusion_v0.NavDiffusionPreprocessor`:

- five real 10 Hz RGB-D frames, with no first-frame padding;
- RGB resized from 640×360 to 160×90, ImageNet-normalized;
- calibrated Z16 depth decoded and registered by `research_data.depth`;
- invalid depth filled at 10 m, clipped/scaled, nearest-neighbor resized, then
  standardized with train-only mean `0.5860419118` and standard deviation
  `0.3728206109`;
- local 2-D mission goal divided by 10 m;
- 32 executed future positions at 0.1 s spacing, normalized by ±2.5 m.

The disposable derived cache lives at `runs/cache/navdiffusion_v0/`. It is
hash-keyed, ignored by Git, and may be rebuilt with:

```bash
make navdiffusion-v0-data
```

## Model and training

The project-owned plain-PyTorch architecture follows the audited
ForestNavigation design at revision
`0b29c399754f510499bfe9cc9d592cba215a7161`:

```text
5 × 4 × 90 × 160 RGB-D
  -> torchvision EfficientNet-B0 ImageNet encoder
  -> 256-d frame embeddings + goal token
  -> 4-layer / 4-head Transformer
  -> Conditional 1-D U-Net [64, 128, 256]
  -> 10-step squared-cosine DDPM
  -> 32 × 2 local trajectory
```

The RGB stem exactly copies ImageNet weights; the fourth depth channel starts
at zero. Native EfficientNet BatchNorm is preserved. The network has 12,675,270
trainable parameters. Training used AdamW, bf16, effective batch 32, five
warmup epochs and cosine decay. It stopped at epoch 120; epoch 100 was selected:

- open-loop validation ADE: `0.090123 m`;
- open-loop validation FDE: `0.169597 m`;
- validation diffusion loss: `0.006512`.

Sanity and smoke commands remain reproducible, but their mutable reports now
write under ignored `runs/` rather than the frozen model directory:

```bash
make navdiffusion-v0-single-batch
make navdiffusion-v0-overfit
make navdiffusion-v0-train
make navdiffusion-v0-smoke
make navdiffusion-v0-ros-smoke
```

## Checkpoint and ROS adapter

`models/trained/navdiffusion_v0/best.pt` is 142,598,595 bytes and has SHA256:

```text
7b3bca67af26c2d839555e9b27a7a420762b572fa88664a227986174d9a68ae0
```

The format embeds model, architecture and preprocessing metadata. The ROS
backend is selected with `model_backend=mns_v0`; legacy ForestNavigation
checkpoint compatibility remains separate. One model sample publishes the
full 32-point diagnostic trajectory while Pure Pursuit consumes its first
eight points.

## Final closed-loop result

The frozen matching-domain system used the accepted Research Forest,
TerrainSurfaceQuery, DIABLO standing surrogate, dual-camera D435i simulation,
shared depth registration, Pure Pursuit and Safety/command arbitration. Hydra
was disabled.

| Evaluation | Success | Conservative collisions | Other failure |
| --- | ---: | ---: | ---: |
| Validation | 4/10 | 6 | 0 |
| One-pass held-out test | 3/10 | 6 | 1 terrain exit |

All six held-out-test collisions were preceded by unsafe full-horizon and
unsafe first-eight control predictions. The moderate, high-density test scene
accounted for five of six test collisions, but environment factors co-vary and
the sample is too small for a causal claim. This is sufficient evidence that
V0 must not control the real DIABLO.

Dataset V0 test was first opened for this final V0 evaluation. Any V1 design
informed by these results must use new held-out scenes for a genuinely untouched
final test.

## Active evaluation path

Gate A/B selection, V0 readiness prose, hard-coded repeatability checks and the
one-time sealed-test workflow are historical. Future navigators use the shared
Research Navigation evaluator:

```bash
make navigation-eval-smoke
make navigation-eval
```

The evaluator provides held-out split safety, scene batching, strict episode
reset, conservative collision checks, timeout/stall handling, prediction-risk,
Safety, progress, clearance, cross-track and inference metrics, atomic reports
and lightweight review plots. Raw traces remain ignored under
`runs/navigation_evaluation/`.
