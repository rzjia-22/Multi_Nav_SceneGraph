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

Only `train_scene_000` and `train_scene_000_episode_000` have been generated.
Generating the other 69 episodes is intentionally deferred for human review.

## Research Forest

The Research Forest is separate from `config/simulation/forest.yaml`, which
remains the fast ROS/Hydra integration regression scene. The research scene is
24×24 m: large enough for several distinct 3–10 m tasks but cheaper to render
than the old 40×40 m prototype.

`research_data/forest.py` uses one NumPy PCG64 stream per scene seed. Terrain
waves, tree positions/types/radii/heights/scales/yaw, ground appearance and
lighting are all derived from that stream or the selected profile. Generated
`scene.yaml` contains every realized parameter and a content hash;
`scene.usda` is a deterministic, diffable scene representation. Rebuilding the
same profile and seed must reproduce both files exactly.

V0 factors are deliberately limited to flat/gentle/moderate terrain,
bare-soil/grass ground, normal/bright/dim lighting and low/medium/high tree
density with two procedural broadleaf shapes. Tree trunks are both rendered
geometry and collision sources. Semantic classes remain `unknown`, `ground`,
`tree_trunk`, `foliage`, `robot` and `other`.

No external vegetation asset is redistributed, so the preview has no asset
license or availability dependency. It was informed by ForestNavigation's
Isaac Lab TerrainImporter, terrain scale and trunk collision approach, but
does not copy its prototype script, unseeded random calls, hard-coded assets or
experimental execution loop.

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

## Preview evidence

`train_scene_000` uses seed 41001, gentle grass terrain, normal lighting and
42 medium-density trees. It starts near (-0.86, 4.72), ends near
(-6.99, 2.80), and has a blocked 6.42 m direct line. The planned detour is
7.187 m. Isaac executes 7.051 m, ends 0.170 m from the goal, and does not
intersect a conservatively checked trunk.

The 12.22 s episode has 122 RGB frames, 122 raw/registered depth frames, 1,224
IMU samples and 612 states. RGB spans 16–252; registered depth has 56.1% valid
pixels. The machine report is
`artifacts/dataset_v0_preview/train_scene_000_episode_000/validation_report.json`.
Review artifacts are `scene_overview.png`, `trajectory_overview.png`,
`camera_samples.png`, `executed_trajectory.csv`, `preview_summary.json` and
the Git-LFS-managed `episode.h5`.

## Reproduction, validation and Git

```bash
make dataset-v0-scene-preview    # CPU scene + expert plan
make dataset-v0-episode-preview  # GPU Isaac execution + collection
make dataset-v0-validate         # split/schema/geometry/data gates
make dataset-v0-visualize        # stable review PNG/CSV/JSON
```

Validation covers the 14/70 plan, bucket counts, leakage, semantics, geometry,
content hashes, exact scene regeneration, timestamps, required calibration,
NaNs, image freshness/range, depth validity, goal error, route limit and
executed footprint clearance.

Source, profiles, manifests, YAML/USDA, CSV/JSON and review PNGs use normal
Git. HDF5, binary USD, videos and future checkpoints are covered by Git LFS.
Generated work, bags, caches and unreviewed runtime outputs remain ignored.

Before expanding to all 70 episodes, human reviewers must decide whether the
0.50 m camera height/FOV, stylized procedural forest, density/lighting and the
single-detour 7.2 m task represent the target field study adequately.

