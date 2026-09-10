# Dataset V0 pilot

Dataset V0 is the model-independent pilot contract for future goal-directed
local visual navigation by a standing-mode DIABLO carrying an Intel RealSense
D435i. It validates the research-forest domain, privileged expert, camera
geometry, storage, and held-out-scene split before bulk collection. It is not a
final training corpus and no navigation model is trained by this workflow.

## Frozen plan

`config/datasets/dataset_v0_manifest.yaml` is authoritative: 10 training,
2 validation, and 2 test scene seeds, with five episodes per scene. A scene ID
belongs to exactly one split. The 70 planned episodes contain 21 short (3–5 m),
28 medium (5–8 m), and 21 long (8–10 m) expert-route buckets. The collector
reads each episode's bucket from this manifest; it never invents a split or
silently changes a bucket at runtime. The first collected
`train_scene_000_episode_000` was invalidated by the strict diagonal-corner
audit described below. The remaining 69 episodes have not been generated.

## One Research Forest path

The Research Forest is independent of `config/simulation/forest.yaml`, which
remains the fast ROS/Hydra integration regression scene. All Dataset V0 scene
review and collection now call the same builder:

```text
scene.yaml
  -> mns_simulation.research_forest_scene.build_research_forest
     -> Isaac Lab TerrainImporter / TerrainGenerator
     -> NVIDIA vegetation USD references
     -> NVIDIA MDL ground
     -> HDR dome and sun
     -> actual mesh TerrainSurfaceQuery
        -> tree Z
        -> review cameras
        -> surrogate Z/normal
        -> camera pose
```

There is no active sine terrain, project-authored formal terrain mesh, or
Cylinder/Sphere tree visual. Hidden conservative cylinder colliders remain an
intentional physical/planning proxy; the camera sees the referenced Blue Berry
Elder and Gray Birch meshes. The source/config/stage snapshot is reproducible,
while NVIDIA assets are referenced rather than redistributed.

The scene schema is version 3. `train_scene_000` remains seed 41001, 24×24 m,
grass, normal light, and medium density. Its 42 unchanged XY realizations are
23 Blue Berry Elder and 19 Gray Birch instances. Its content hash is
`07d75706a35ad4a7e02f696d2ce6ff5e0c1eae4e659651922a96e039e8e158f4`.

## Terrain calibration

`config/research_forests/profiles.yaml` maps only to Isaac Lab 2.3.1 official
height-field generators. `config/research_forests/terrain_calibration.yaml`
records measurements from `TerrainGenerator.terrain_mesh` using seed 41001 and
the 24×24 m extent. These are geometry measurements, not theoretical design
limits:

| Profile | Generator | Elevation range / std | Slope mean / median | p90 / p95 / max |
| --- | --- | --- | --- | --- |
| flat | `HfRandomUniformTerrainCfg`, zero noise | 0 / 0 m | 0 / 0° | 0 / 0 / 0° |
| gentle | `HfWaveTerrainCfg`, amplitude 0.12–0.22 m, 2 waves | 0.312 / 0.078 m | 2.258 / 2.361° | 3.042 / 3.238 / 3.641° |
| moderate | `HfWaveTerrainCfg`, amplitude 0.15–0.27 m, 3 waves | 0.408 / 0.102 m | 4.414 / 4.609° | 6.054 / 6.258 / 6.855° |

The 0.001 m vertical scale on wave profiles prevents quantization plateaus.
Flat, gentle, and moderate are consequently distinct and remain provisional
until measured DIABLO field data are available.

The shared `TerrainSurfaceQuery` interpolates the actual imported USD mesh
triangles and returns both `height(x,y)` and the upward surface normal. The
surrogate's x/y motion follows its navigation yaw projected onto this surface;
its body z-axis follows the normal. Camera pose is composed from that body
frame, the configured DIABLO mount, one deterministic episode-level offset,
and small continuous correlated motion. Terrain tilt is not treated as random
camera vibration.

## DIABLO and D435i assumptions

`config/robots/diablo_standing.yaml` remains a provisional target profile. The
stable kinematic non-holonomic surrogate has a 0.70×0.46 m footprint, 0.42 m
collision-check radius, 0.58 m planning radius, 0.65 m/s forward limit, and
1.0 rad/s yaw limit. It isolates visual-data quality from wheel-legged dynamic
failures. Its dynamics, suspension, slip, geometry, and vibration are not a
high-fidelity DIABLO model. The 0.50 m camera height and all mount values must
eventually be replaced by physical calibration in the profile, not code.

The official capability references remain the Intel
[D400 family datasheet](https://www.intelrealsense.com/wp-content/uploads/2023/03/Intel-RealSense-D400-Series-Datasheet-March-2023.pdf)
and [D435i librealsense note](https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md).
Dataset V0 logs 640×360 RGB and 848×480 depth at an application rate of 10 Hz,
plus clean 100 Hz simulator IMU and 50 Hz state. Rendered intrinsics are read
from Isaac. RGB and depth retain separate FOV/intrinsics and a provisional
extrinsic. Real collection must read each device's intrinsics, distortion,
extrinsic, and depth scale from librealsense.

## HDF5 schema version 2

Each episode is an independent HDF5 file. It remains raw and model-independent;
five-frame histories, resized images, and future-waypoint labels are derived by
offline model preprocessing.

| Group | Stored content |
| --- | --- |
| root attributes | schema/dataset version and complete JSON metadata, including scene hash and collector version |
| `state/` | timestamps, terrain-following pose quaternion, 3D velocities, and executed command |
| `imu/` | timestamps, clean acceleration, and angular velocity |
| `sensors/` | RGB `uint8`, raw depth `uint16`, timestamps, and RGB camera pose |
| `calibration/` | rendered intrinsics, depth-to-RGB extrinsic, resolutions, depth scale, alignment algorithm/version, invalid convention |
| `expert/` | global privileged expert path |

Raw simulated depth is encoded as RealSense-style Z16 with a Dataset V0 scale
of 0.001 m/unit; zero means invalid. This is a simulation storage choice, not a
claim that every physical D435i has the same scale. No value in the configured
0.2–10 m range saturates. RGB remains losslessly compressed `uint8`.

Aligned depth is deliberately absent from the file. `research_data.depth`
reconstructs it from raw Z16 plus calibration using the versioned z-buffer
reprojection. During the formal collection the persisted episode reproduced
the in-memory aligned reference with 100% valid-pixel agreement and zero depth
difference. Z16 quantization measured 0.250 mm mean, 0.475 mm p95, and
0.501 mm maximum absolute error, with zero saturation.

## Superseded pilot and measured cost

The existing pilot remains at
`artifacts/dataset_v0_preview/train_scene_000_episode_000/`, but it is no longer
a valid Dataset V0 episode. The old 8-connected A* checked only the diagonal
destination cell. A strict audit found four smoothed segments that transition
diagonally while one adjacent orthogonal inflated-grid cell is occupied. The
corrected planner permits a diagonal only when both side cells are free, and
the validator now rejects this pilot. Its 70 RGB/depth frames, 703 IMU samples,
351 states and benchmark remain useful implementation evidence only.

The 50,295,537-byte HDF5 took 1.788 s to write. Compressed contributions are
31,046,857 bytes RGB, 19,073,847 bytes depth, 49,208 bytes state/IMU/expert/
calibration, and 125,625 bytes HDF5 metadata overhead. RTX execution took
14.353 s (RTF 0.488) after 27.337 s application startup and 12.625 s scene
load. The machine report in `benchmark_report.json` is authoritative.

Extrapolating this single short episode across the manifest's approximately
455 m gives a deliberately uncertain 6.27 GB point estimate (4.39–10.02 GB).
Naively restarting for every episode projects 78.5 minutes (58.9–137.4), while
loading each of 14 scenes once and executing five plans projects 41.2 minutes
(30.9–72.1). A local Git LFS checkout plus its object copy is about 12.53 GB at
the point estimate; 25.1 GB free is recommended. These are planning estimates,
not promises: depth/RGB compressibility and route visibility differ by scene.

On the RTX 4060 Laptop, peak VRAM was 5,217/8,188 MiB, process RAM 10,101 MiB,
GPU utilization 5.7% average/19% peak, 46.5°C average/48°C peak, and
10.1 W average/12.6 W peak during the sampled interval. The current conclusion
is **RTX 4060 Laptop: SUFFICIENT** for a single-worker Dataset V0 collection;
no server or parallel Isaac workers are justified by this pilot.

The earlier primitive-domain preview is rejected and removed from current
`main`; it remains recoverable at Git commit `85d3fe6` and is not supported by
the current schema.

## Commands and validation

```bash
make dataset-v0-calibrate-terrain       # measure all three official profiles
make dataset-v0-scene-preview           # deterministic YAML + schematic
make dataset-v0-capture-scene-review    # four fixed RTX views
make dataset-v0-view-scene              # interactive Isaac, no navigation
make dataset-v0-episode-benchmark       # blocked until episode_000 regeneration is authorized
make dataset-v0-validate                # scene/split/runtime/HDF5/registration
```

The collector accepts repeated `--episode-id` arguments within one Isaac scene
lifecycle, which is the future scene-batched path. No Make target currently
generates all 70 episodes. Validation checks conservative no-corner-cut planned
segments before accepting HDF5 evidence, in addition to the 14/70 scene-level
split, manifest bucket, deterministic scene specification, actual terrain,
hashes, shared builder, HDF5 v2, Z16 calibration, offline registration,
timestamps, terrain-following pose, collision-free execution and non-stale RGB.

Sources, YAML/JSON/CSV, the compact USDA, and review images use ordinary Git.
The formal HDF5 is Git LFS. NVIDIA assets remain external URI references.
Caches, Isaac logs, ROS bags, and temporary collection output are ignored.
