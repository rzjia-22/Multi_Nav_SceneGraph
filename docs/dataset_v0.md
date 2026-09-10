# Dataset V0 pilot

Dataset V0 is the pilot data contract for future goal-directed local visual
navigation on a standing-mode DIABLO carrying an Intel RealSense D435i. It is
not a final training corpus. Its purpose is to review the simulation domain,
expert generation, camera geometry and model-independent storage before
authorizing the remaining episodes.

## Fixed plan and split

`config/datasets/dataset_v0_manifest.yaml` is the only authoritative plan.
It assigns immutable IDs and seeds to 14 scenes and five episodes per scene:

| Split | Scenes | Episodes |
| --- | ---: | ---: |
| train | 10 | 50 |
| validation | 2 | 10 |
| test | 2 | 10 |

The split is scene-level. A validator fails if one `scene_id` appears in more
than one split. The 70 planned routes contain exactly 21 short (3–5 m), 28
medium (5–8 m) and 21 long (8–10 m) targets. The final bucket is determined by
expert path length; a sampled task is rejected if its actual planned route is
outside its bucket or materially exceeds 10 m.

Only `train_scene_000` has been rebuilt as the formal visual-scene candidate.
All episode generation is blocked until this scene passes human visual review.
The earlier `train_scene_000_episode_000` is retained as historical evidence
but is explicitly rejected for training in its `review_status.yaml`.

## Research Forest

The Research Forest is separate from `config/simulation/forest.yaml`, which
remains the fast ROS/Hydra integration regression scene. The research scene is
24×24 m: large enough for several distinct 3–10 m tasks but cheaper to render
than the old 40×40 m prototype.

`research_data/forest.py` uses one NumPy PCG64 stream per scene seed, but now
produces only a scene specification. It does not draw terrain or trees. The
shared Isaac builder in `mns_simulation.research_forest_scene` maps that
specification to `TerrainImporterCfg` / `TerrainGeneratorCfg`, official height
field generators, `UsdFileCfg` vegetation references, an MDL ground material,
and HDR dome plus sun lighting. `scene.yaml` records all selected IDs,
placements, scales, yaw, collision proxies, semantic policy and version pins.
The compact `scene.usda` snapshot is exported from the authored Isaac root
layer and preserves external asset references rather than flattening them.

V0 factors remain limited to flat/gentle/moderate terrain, bare-soil/grass
ground, normal/bright/dim lighting and low/medium/high tree density. The formal
tree visuals are NVIDIA simulation-ready `Blue_Berry_Elder.usd` and
`Gray_Birch.usd`; no Cylinder or Sphere is visible to a camera. Invisible
conservative cylinders remain as collision proxies. The two assets expose
stable `trunk` and `leaves` prims, so roots are tagged `vegetation` while those
children remain `tree_trunk` and `foliage`. The parent fallback maps to the
existing Hydra-compatible `other_object` ID 7; `robot` remains ID 6. No label
number was reassigned.

`config/research_forests/assets.yaml` is the asset registry. It references the
Isaac Sim 5.1 NVIDIA cloud asset root, `Grass_Countryside.mdl`, `Dirt.mdl`, and
the Kloofendal Poly Haven HDR shipped through the Isaac asset library. NVIDIA
binaries and textures are not copied into this repository. Their stable logical
IDs, source, expected scale, collision and semantic policy are versioned.

## Robot target and surrogate

`config/robots/diablo_standing.yaml` is a target profile, not a DIABLO hardware
specification. The preview uses a stable kinematic, non-holonomic box surrogate
with a 0.70×0.46 m provisional footprint, 0.42 m collision-check radius,
0.58 m planning radius, 0.65 m/s forward limit and 1.0 rad/s yaw limit. It
separates visual-navigation data quality from wheel-legged locomotion failure.

It differs from real DIABLO dynamics, suspension, appearance, slip and camera
vibration. Camera height 0.50 m, its allowed 0.40–0.60 m range, the 0.40 m
forward mount, motion limits and footprint are provisional simulation values.
They must be replaced with physical measurements only in the profile.
Episode-level mount ranges and continuous vertical/pitch/roll motion are
configured; the preview uses small correlated amplitudes, never per-frame
independent randomization.

## D435i navigation profile

`config/sensors/d435i_navigation_v0.yaml` separates device capability from its
10 Hz application profile. Sources are the official
[D400 family datasheet](https://www.intelrealsense.com/wp-content/uploads/2023/03/Intel-RealSense-D400-Series-Datasheet-March-2023.pdf)
and [D435i librealsense note](https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md).
Relevant capabilities are maximum 1920×1080/30 Hz RGB with nominal 69°×42°
FOV, maximum 1280×720/90 Hz depth with nominal 87°×58° HD FOV, and a 6DoF
IMU (accelerometer 62.5/250 Hz; gyro 200/400 Hz). The optical/IMU convention
is x right, y down, z forward.

V0 logs 640×360 RGB and 848×480 raw metric depth at 10 Hz. Isaac 5.1 uses
square-pixel pinholes, so these aspect ratios approximate the nominal FOV pair;
exact rendered matrices are read from Isaac and stored. Raw depth is z-buffer
reprojected through separate intrinsics and a provisional extrinsic into a
640×360 RGB-aligned image. The 15 mm translation is an assumption: real
collection must read factory calibration and distortion from librealsense.

The preview records one clean 100 Hz simulator IMU stream. This is an
application schema choice, not a D435i hardware mode. A future real collector
should preserve native asynchronous timestamps. `d435i_randomized` remains an
extension point for dropout, range noise, exposure and outdoor stereo failure;
no complex noise model is synthesized here.

## Expert and raw schema

The deterministic privileged-map expert is 8-connected 0.10 m grid A* with
conservative trunk inflation and line-of-sight smoothing. A non-holonomic Pure
Pursuit controller executes the route at 50 Hz inside Isaac. This is training
demonstration generation, not a deployment planner. The schema reserves
`nominal_expert` and future `perturbed_recovery`; only nominal is generated.

Each episode is an independent HDF5 file:

| Group | Content |
| --- | --- |
| root attributes | schema/dataset version and complete JSON metadata |
| `state/` | timestamps, pose, velocities and executed command |
| `imu/` | timestamps, linear acceleration and angular velocity |
| `sensors/` | timestamps, RGB, raw depth, RGB-aligned depth and optical pose |
| `calibration/` | actual intrinsics and provisional RGB-depth extrinsic |
| `expert/` | global privileged expert path |

This is not fixed to the legacy five frames, 128×96 input or 32 waypoints.
`config/models/navigation_input_v0.yaml` is a model-agnostic preprocessing
template; future model architecture, history, resize, normalization, horizon
and training parameters remain version-controlled configuration.

## Current scene-review evidence

`train_scene_000` keeps seed 41001, a 24×24 m gentle grass terrain, normal
lighting and 42 medium-density trees. Its fixed realization contains 23 Blue
Berry Elder and 19 Gray Birch instances. Tree roots are sampled from the actual
Isaac-imported height field; scale and yaw are recorded in `scene.yaml`.

The formal review directory is
`artifacts/dataset_v0_scene_review/train_scene_000/`. `scene_layout.png` is
explicitly a schematic XY/collision overview. `isaac_aerial.png`,
`isaac_ground_view.png` (0.5 m camera height), `isaac_mid_height.png`, and
`isaac_tree_closeup.png` are actual 1280×720 Isaac RTX outputs. The machine
report records zero missing registry assets, a 17.68 s scene load, about 5,132
MiB VRAM, and 15.29 rendered frames/s during the final fixed-view capture.

The older episode directory remains unchanged except for a rejection marker.
Its primitive-domain camera image, expert path and HDF5 are not valid Dataset
V0 training evidence and must not be regenerated before this visual gate.

## Reproduction, validation and Git

```bash
make dataset-v0-scene-preview         # deterministic spec + XY schematic only
make dataset-v0-capture-scene-review  # four fixed Isaac RTX review images
make dataset-v0-view-scene            # interactive non-headless Isaac viewer
make dataset-v0-validate              # split/spec/stage/reference gates
```

`make dataset-v0-view-scene` must be run from the NVIDIA host's X11 desktop
session. It opens the same builder and `train_scene_000` used for fixed capture,
starts no robot, ROS, navigation, collector or Hydra process, and remains open
for free viewport inspection. The Xauthority cookie is mounted read-only.

Current validation covers the frozen 14/70 plan, bucket counts, leakage,
semantic concepts, registry membership, the official terrain builder,
primitive-visual exclusion, content hash, deterministic scene-spec
regeneration and the Isaac-authored stage snapshot. Episode validation remains
implemented but is not part of the current acceptance gate.

Source, profiles, manifests, YAML/USDA, CSV/JSON and review PNGs use normal
Git. HDF5, binary USD, videos and future checkpoints are covered by Git LFS.
Generated work, bags, caches and unreviewed runtime outputs remain ignored.

Before any episode is regenerated, human reviewers must decide whether the two
tree assets resemble the intended woodland, density and corridor openness are
appropriate, the grass/terrain is credible, the 0.5 m view has the right scale,
and normal lighting is neither overexposed nor too dark. D435i geometry and
task difficulty return to review only after the scene itself passes.
