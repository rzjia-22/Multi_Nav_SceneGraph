# Development status

This is the sole authoritative status document. Last updated 2026-09-11.

## Current milestone

Phase 1 and the functional Phase 2 acceptance now pass with real Isaac Sim /
Isaac Lab forest data on the current NVIDIA host. This is no longer a
synthetic-only integration: articulated Go2 locomotion, kinematic UAV motion,
registered RGB/depth/semantic sensing, ROS 2 navigation and independent online
Hydra pipelines ran concurrently and saved inspectable DSG, mesh, trajectory
and timing artifacts.

The four-robot run is resource-limited rather than wall-clock real-time on this
RTX 4060 Laptop / 16 GB RAM machine. All four cameras remain exactly 10 Hz in
simulation time, but wall throughput is about 2.4 Hz (RTF about 0.24). This is
reported as a performance warning, not hidden as a functional failure.

Dataset V0 collection is complete. The accepted Research Forest visual domain,
calibrated terrain profiles, D435i-like sensor contract, DIABLO surrogate
profile, and scene-level split remained frozen. Planner v2 applies one strict
no-corner-cut transition contract to A*, LOS smoothing, and final validation.
All 70 exact plans passed preflight, and all 70 scene-batched RTX episodes pass
HDF5 v2, provenance, sensor, execution, and aggregate validation.

## Dataset V0 milestone state

| Capability | State | Evidence |
| --- | --- | --- |
| Dataset plan | source complete, validated | 10/2/2 scene split; 21/28/21 short/medium/long; no leakage |
| Research Forest specification | deterministic validated | scene schema v3; seed 41001; one shared builder; hash `07d75706...` |
| Research Forest rendering | GPU Isaac validated, human accepted | 23 Blue Berry Elder + 19 Gray Birch, Grass MDL, HDR sky; four updated RTX views; zero missing assets |
| Terrain profiles | GPU Isaac geometry validated | flat median/p90 0/0°; gentle 2.361/3.042°; moderate 4.609/6.054° |
| DIABLO profile | provisional V0 assumptions | non-holonomic surrogate; mount/footprint/limits isolated in config |
| D435i profile | official capabilities documented; HDF5 v2 validated | 640×360 RGB, 848×480 Z16 depth at 0.001 m scale; offline registration exact against runtime reference |
| Expert/recorder | GPU Isaac validated | planner v2; both orthogonal cells required for diagonals; exact plan/YAML/HDF5 provenance enforced |
| Historical primitive preview | removed from current tree | rejected implementation and artifacts remain available at commit `85d3fe6` only |
| Scene visualization | GPU Isaac validated | aerial, 0.5 m ground, mid-height and close vegetation RTX views; interactive X11 viewer reached ready state |
| Scene/episode validation | PASS | 70/70 strict plans and 70/70 finalized HDF5 v2 episodes; no split leakage |
| Bulk generation | complete | train 50, validation 10, test 10; 21/28/21 short/medium/long |

## Milestone state

| Milestone | State | Real runtime evidence |
| --- | --- | --- |
| A — version freeze and architecture | complete | pinned images, source graph, packages, contracts and docs |
| B — Isaac Go2 and sensor bridge | GPU Isaac validated | articulated Go2 plus RGB/depth/semantic/CameraInfo/odom/TF/clock |
| C — single Hydra integration | GPU Isaac validated | direct live input, non-empty DSG/mesh/trajectory and clean save |
| D — Go2 locomotion backend | GPU Isaac validated | ROS Twist → 48-value observation → actor → 12 joint targets at 50/200 Hz |
| E — Coverage + follower + Hydra | GPU Isaac validated | obstacle-aware route, stable movement and concurrent mapping |
| F — Diffusion + Hydra | GPU Isaac validated | five real RGB-D frames, GPU inference, follower motion and completed goal |
| G — Nav2 + Hydra | GPU Isaac validated | live Isaac depth point cloud/costmaps, action success and mapping |
| H — Phase 1 acceptance | PASS | all three replaceable navigators validated against real Isaac sensors and RL motion |
| I — UAV migration | GPU Isaac validated | 42 m aerial coverage, correct nadir RGB-D/semantics and meaningful Hydra reconstruction |
| J — 2 Go2 + 2 UAV | PASS, resource-limited RTF | one world, four isolated namespaces/TF trees/Hydra outputs; no estimated input gaps |
| K — cleanup and handoff | complete | unified commands, artifact inspector, ignored outputs and updated authority docs |

## Implemented capabilities

- Eight ROS 2 packages and one configuration-driven launch generator for both
  single- and multi-robot rosters.
- Per-robot registered RGB, metric depth, integer semantics, color/depth
  CameraInfo, odometry and globally unique TF, plus one shared simulation clock.
- Replaceable Coverage, Diffusion and Nav2 strategies behind the same mission,
  Twist, safety and motion-backend boundaries.
- Original audited NavDiffusion and Go2 actor checkpoints with hash verification
  and tensor-only loading; neither model was retrained.
- A deterministic collision-enabled semantic forest with ground, trunks and
  foliage. Known-area Coverage uses the same configured tree centres for its
  obstacle-aware A* connectors.
- Real Go2 actor execution at 50 Hz over 200 Hz GPU PhysX. Render selection is
  10 Hz without changing the one-tick physics/render timestep.
- Batched Isaac Lab cameras for multiple instanceable Go2/Crazyflie assets,
  while every ROS publisher, navigator and Hydra process remains per robot.
- Nadir UAV cameras attached to the moving Crazyflie body link. The planar
  forward-depth safety node is intentionally Go2-only; the nadir mapping camera
  is not misused as an aerial collision sensor.
- Independent Hydra robot IDs, map frames, DSG topics and output directories.
  There is no global DSG, fusion, loop closure or entity reconciliation.
- Machine-readable observers distinguish simulation-time sensor frequency from
  wall throughput and check decoded payloads, image-time TF, motion, mission
  state, DSG throughput and estimated timestamp gaps.
- `tools/inspect_hydra_artifacts.py` summarizes finalized graphs, semantic mesh
  geometry and trajectories without creating a GUI or committing large runs.

## Tests and runtime evidence

- `make build`: all 8 ROS packages built; 10 package tests passed, with zero
  errors, failures or skips.
- `make validate`: repository validation and 23 pure Python tests passed.
- Dataset V0 scene capture: Isaac Lab regenerated the 24×24 m terrain, loaded
  the unchanged 23 Blue Berry Elder and 19 Gray Birch placements, resolved the
  Grass MDL and Kloofendal HDR, and reported zero missing assets. The updated
  gentle mesh has 0.312 m elevation range, 2.361° median and 3.042° p90 slope.
  Scene load took 11.80 s; 1280×720 fixed-view rendering averaged 19.70 FPS and
  used about 5,132 MiB of 8,188 MiB VRAM.
- Dataset V0 Planner v2 tests cover all four diagonal directions, occupied-side
  rejection, grid boundaries, long LOS segments, smoothing, and serialized
  plan revalidation. The exact 70/70 plan preflight passed with the frozen
  21/28/21 route buckets.
- The regenerated `train_scene_000_episode_000` strict plan hash is
  `e83cdb9b...`; it planned 3.876 m, executed 3.736 m, reached 0.233 m goal
  error, and passed collision/provenance/sensor/storage checks. The rejected
  predecessor remains in history at `67a6b81` only.
- HDF5 v2 Z16 at 0.001 m/unit had 0 saturation,
  0.250/0.475/0.501 mm mean/p95/max absolute error. Offline registration had
  100% valid-pixel agreement and zero depth difference from the in-memory
  reference across collection validation.
- Dataset V0 final totals: 70 episodes, 458.516 m planned / 445.278 m executed,
  779.88 s simulated, 7,781 RGB/depth frames, 78,155 IMU samples, 39,064
  states, and 4,699,759,237 HDF5 bytes. Reported collection-session wall time
  was 3,686.80 s. Mean/min RTF was 0.472/0.319; peak VRAM 5,345 MiB, peak
  process RSS 11,202 MiB, and peak GPU temperature 51°C. No retries or
  memory-leak sessions were reported. **RTX 4060 Laptop: SUFFICIENT.**
- `make dataset-v0-view-scene` reached `MNS_RESEARCH_FOREST_READY` through the
  host Xauthority path, opened the non-headless Isaac renderer and was then
  stopped manually. It launched no robot, ROS graph, navigator or collector.
- `make gpu-preflight`, `make isaac-compatibility`, `make isaac-minimal`: RTX
  4060 Laptop GPU, 8,188 MiB VRAM, driver 550.144.03, CUDA container and Vulkan
  pass; frozen Isaac Lab 2.3.1 GPU PhysX steps and shuts down cleanly.
- Real Go2 sensor gate: 10 Hz simulation-time RGB8, 32FC1 metric depth, 16UC1
  labels and CameraInfo; 200 Hz odom; valid timestamped TF and clock.
- Real articulated motion gate with cameras: 2.397 m forward, 0.985 rad turn,
  stable upright base, explicit stop and 0.3 s command deadman pass.
- Coverage + Hydra observer (20 s wall): 114 registered image sets, 2,269 odom,
  11 backend DSG updates, 3.265 m displacement, 249 mapping inputs, no estimated
  gaps and complete TF. Final graph had 269 nodes/308 edges; mesh had 49,557
  vertices with ground, tree-trunk and foliage structure.
- Diffusion + Hydra: history 5/5, 14 GPU plans (last measured 58.7 ms), real
  follower/actor motion to the configured goal, mission `complete`; a concurrent
  15 s observer saw 95 image sets, 1,885 odom, 9 DSG updates and no gaps.
- Nav2 + Hydra: launch-time action rejection is retried until lifecycle active;
  the accepted goal succeeded after real articulated motion. Isaac depth is
  converted to a 320-wide point cloud at about 6.3 wall Hz in the single-robot
  run, and both local/global costmaps subscribe to the absolute namespaced
  point-cloud topic. Final Hydra graph/mesh/trajectory saved on exit.
- Single UAV sensor gate: 10 Hz simulation-time images, correct 5.92 m ground
  depth from 6 m altitude, and a sample containing ground 38,885, trunk 2,367
  and foliage 35,548 pixels. A 20 s observer recorded 2.79 m displacement, 15
  DSG updates and no gaps during the longer mission.
- Final single-UAV artifacts: 89.6 s trajectory over the configured 42 m route;
  177 DSG nodes/281 edges and 27,990 mesh vertices. Ground median z is 0 m,
  trunk median 1.05 m and foliage median 4.25 m; aerial forest structure is
  meaningful rather than merely a non-empty DSG.
- Final Phase 2 sensor gate: all four robots passed encodings, intrinsics,
  timestamped TF and exact 10 Hz simulated image / 200 Hz odom rates. The
  15-second system observer passed for all four robots with concurrent motion,
  four running Hydra pipelines, four backend DSG streams and zero estimated
  timestamp gaps.
- Final Phase 2 resources: about 4.5 GB VRAM; Isaac about 3.0 GB RAM / 169% CPU;
  robotics about 6.4 GB RAM / 902% CPU. Wall image throughput was 2.4 Hz for
  each robot (RTF 0.24), identifying CPU/RAM concurrency as the practical host
  bottleneck rather than namespace, TF, DDS or GPU-memory corruption.
- All Phase 2 processes exited code 0. Four finalized outputs contained
  respectively 386/158/149/152 DSG nodes and 68,972/31,016/26,901/25,463 mesh
  vertices for `go2_1`, `go2_2`, `uav_1`, `uav_2`.

Runtime artifacts live under ignored `runs/` directories. Summarize one with:

```bash
MNS_HYDRA_DIR=runs/<run-id>/<robot>/hydra make inspect-hydra
```

## Run commands

Run bringup and its observer in separate terminals:

```bash
make gpu-preflight
make isaac-compatibility
make isaac-minimal
make models

MNS_RUN_ID=coverage-real make phase1
MNS_ACCEPTANCE_DURATION=20 make accept-phase1

MNS_RUN_ID=diffusion-real make phase1-diffusion
MNS_NAVIGATOR=nav2 MNS_RUN_ID=nav2-real make phase1

MNS_RUN_ID=uav-real make uav-mapping
MNS_ACCEPTANCE_DURATION=20 make accept-uav

MNS_RUN_ID=phase2-real make phase2
MNS_ACCEPTANCE_DURATION=20 make accept-phase2
```

Stop bringup with Ctrl-C or `docker compose stop`; SIGINT reaches Python PID 1
and Hydra saves all artifacts before both containers exit.

## Known remaining issues

- The current 8 GB VRAM / 16 GB RAM laptop cannot run the four-camera/four-Hydra
  graph at wall-clock real time with the default 320×240, 10 Hz configuration.
  Functional simulated rates are exact; use a stronger host or deliberately
  tune sensor resolution/rate for real-time performance.
- Go2 meshes show a negative ground tail during body tilt (ground median near
  -0.1 m, p99 near 0 m, minimum as low as -2.45 m in the longest Phase 2 run).
  Scale, dominant ground plane, trunks and canopy remain coherent. Further
  calibration/TSDF tuning should use recorded evidence rather than changing
  Hydra algorithms blindly.
- `PipelineStatus.dropped` estimates source timestamp gaps; pinned Hydra does
  not expose its internal queue-drop counter. Per-frame mapping latency and
  explicit physical contact/safety-override counters remain observability work,
  not prerequisites hidden as completed measurements.
- The full initial-plus-residual Go2 coverage route has not been run to natural
  exhaustion. Initial known-map obstacle avoidance and online sensor coverage
  are validated; unknown-area exploration remains deliberately out of scope.

## Architecture decisions

- Ubuntu 22.04 host; ROS 2 Jazzy in Ubuntu 24.04 robotics containers.
- Isaac Sim 5.1.0 / Isaac Lab 2.3.1 is the only simulator combination.
- Commit-pinned Hydra/Hydra-ROS and dependencies; no ROS 1 path.
- Isaac's Jazzy ROS bridge publishes standard ROS messages only. Navigation and
  mapping do not import Isaac APIs.
- Go2 RL locomotion is a replaceable motion backend, never part of Diffusion.
- One Hydra pipeline per robot is the Phase 2 isolation boundary.
- UAVs remain kinematic sensor platforms; a real flight stack is future adapter
  work, not part of this phase.

## ForestNavigation migration state

Migrated and reorganized: Go2 Isaac assets and actor inference, RGB-D history
and NavDiffusion inference, pure pursuit, zigzag/connected/sensor/residual
coverage, A* connectors, mapped-depth safety, stall recovery, Nav2 handoff,
kinematic UAV behavior and multi-team partitioning.

Intentionally excluded: ROS 1 handoff/conversion/validation, 1000 m reliability
and static reconstruction benchmarks, training, historical wrappers, duplicate
renderers/assets, backup/final scripts and conflicting environment freezes.
Attribution, modifications and model hashes are in
`docs/forestnavigation_migration.md` and `THIRD_PARTY_NOTICES.md`.

## Future deployment boundaries

Real robots still require calibrated/time-synchronized sensor adapters, a
Unitree high-level locomotion adapter and a UAV flight-stack adapter. The ROS
sensor and `cmd_vel_safe` contracts do not change.

Future global-DSG research would additionally need inter-robot transforms,
loop closure, fusion policy, entity reconciliation and a global graph owner.
Those modules are intentionally not implemented.
