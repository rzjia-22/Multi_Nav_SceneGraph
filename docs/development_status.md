# Development status

This is the sole authoritative status document. Last updated 2026-09-09.

## Current milestone

GPU preflight and the deterministic real Isaac forest sensor milestone pass on
the current host. Frozen Isaac Lab 2.3.1 executes Vulkan rendering and GPU
PhysX, and a real Go2 view contains ground, tree-trunk and foliage classes in
decoded RGB, metric depth and integer semantics with timestamped CameraInfo,
odom, TF and clock. The next boundary is the articulated RL motion backend,
followed by real forest Coverage + Hydra.

This is still not represented as full Phase 1/2 completion. Diffusion, Nav2,
UAV aerial reconstruction and the four-robot topology remain synthetic-only
until their real Isaac runs and saved Hydra artifacts are inspected.

## Milestone state

| Milestone | State | Evidence / boundary |
| --- | --- | --- |
| A — version freeze and architecture | complete | pinned images, Hydra graph, ROS packages, contracts and docs |
| B — Isaac Go2 and sensor bridge | GPU Isaac validated for one kinematic Go2 in forest | 2.3.1 GPU PhysX PASS; RGB/depth/four semantic classes and timestamped TF inspected |
| C — single Hydra integration | complete in synthetic | real Hydra consumes all four image inputs/TF and publishes non-empty DSG updates |
| D — Go2 locomotion backend | source and checkpoint smoke complete; physics pending | safe 48-to-12 actor forward pass succeeds |
| E — Coverage + follower + Hydra | complete in synthetic | live Twist motion, sensor coverage, Hydra and safety run concurrently |
| F — Diffusion | complete in synthetic | original checkpoint generates `(8,2)` paths at about 2 Hz and drives the common follower |
| G — Nav2 | complete in synthetic | namespaced lifecycle stack accepts and completes the configured goal |
| H — Phase 1 acceptance | synthetic pass; Isaac sensors pass; navigation/Hydra pending | real Go2 sensor observer has zero failures |
| I — UAV migration | source complete; synthetic pass; Isaac imagery pending | UAV namespace, kinematic motion and mapping isolation verified |
| J — 2 Go2 + 2 UAV | synthetic pass; Isaac pending | four navigation and four Hydra instances run simultaneously without TF collisions |
| K — cleanup and handoff | complete except GPU evidence | authoritative docs, unified commands, ignored outputs and repeatable observers |

## Implemented capabilities

- Eight ROS 2 packages with configuration-driven Phase 1 and Phase 2 launch.
- One standard per-robot sensor contract: RGB, metric depth, integer semantic,
  CameraInfo, odometry, TF and shared simulation clock.
- Stable prefixed frames and namespaces for `go2_1`, `go2_2`, `uav_1`, `uav_2`.
- One independent Hydra process, map frame, numeric robot ID, DSG topic and run
  directory for every mapping-enabled robot.
- Replaceable Coverage, Diffusion and Nav2 navigators behind the same mission,
  Twist and motion-arbiter interfaces.
- Connected and zigzag route planning, A* connectors, pure pursuit, mapped
  depth safety, stall recovery, sensor coverage and bounded residual passes.
- Original audited Diffusion architecture/checkpoint and Go2 actor with
  hash-verified assets and tensor-only safe loading.
- Shared Isaac runtime with a deterministic collision-enabled semantic forest,
  two articulated Go2 instances, two kinematic UAV
  platforms, 200/50/10 Hz physics/policy/sensor scheduling and GT semantic
  remapping. Single-Go2 kinematic sensing is GPU-executed; RL and multi-robot
  runtime evidence are still pending.
- Machine-readable runtime acceptance for decoded sensor payloads, simulation
  and wall rates, image-time TF, motion, TF,
  mission state, DSG updates, mapping state and input timestamp gaps.
- Optional MCAP topic policy and isolated ignored runtime output directories;
  bags are not part of the online mapping path.

## Tests and runtime evidence

The latest completed results are:

- `make validate`: repository validation plus 17 pure Python tests passed,
  including the configured forest's obstacle-aware coverage route.
- `make gpu-preflight`: RTX 4060 Laptop GPU, 8,188 MiB VRAM, driver
  550.144.03; host and CUDA-container probes pass; graphics capabilities are
  `all`.
- `make isaac-minimal`: frozen Isaac Lab 2.3.1 used `cuda:0`; a rigid body fell
  from 0.998 m and settled at 0.100 m after 120 steps; exit code 0.
- Real Isaac single-Go2 forest acceptance, 20 seconds: 128 registered RGB,
  depth, semantic and CameraInfo samples at exactly 10.0 Hz simulation time
  (6.40 Hz wall throughput), 2,563 odometry samples at 200 Hz simulation time,
  advancing clock, complete TF and transform lookup at the image timestamp.
  Payloads were RGB8, 32FC1 metres and 16UC1 class IDs. Finite depth covered
  57.4% of the image with 1.39/1.95/9.51 m minimum/median/maximum; class counts
  included unknown 32,729, ground 27,664, tree trunk 16,256 and foliage 151.
- The same forest runtime completed 5,000 bounded steps, emitted a
  machine-readable PASS result and exited with code 0. This is real runtime
  shutdown evidence, not a source-only path.
- `make build`: all 8 ROS packages built; 9 package tests passed with no
  errors, failures or skips.
- `make test-models`: real Diffusion checkpoint returned finite `(8,2)` output;
  real Go2 actor returned finite `(2,12)` output on CPU.
- Robotics image rebuilt from the pinned source graph; ML image rebuilt and
  `pip check` reported no broken requirements.
- Phase 1 Coverage + Sensor Coverage + Hydra, 8-second observer: RGB/depth/
  semantic/CameraInfo each 80 frames, depth 9.997 Hz, odom 401 frames, eight
  backend DSG updates, 5.58 m synthetic displacement, mapping `running`, no TF
  gaps, estimated input drops 0.
- Phase 1 Diffusion + Hydra: five-frame RGB-D history ready, path publication
  about 2 Hz, robot advanced from x=-6.0 to about -4.67 toward its goal; Hydra
  received 819 RGB frames, published at about 1 Hz and reported zero gaps.
- Phase 1 Nav2: lifecycle stack reached the configured goal and reported
  `complete`; final x was about -4.835 for goal x=-4.5, inside the configured
  0.35 m tolerance.
- Final Phase 2, 10-second observer: each robot delivered 100–101 RGB/depth/
  semantic/CameraInfo frames at 9.99–10.09 Hz, 486–490 odometry frames, 6–7
  backend DSG updates and 4.78–5.37 m synthetic displacement. All four had
  complete prefixed TF, running independent missions/mapping and zero estimated
  input gaps.
- Four-Hydra shutdown regression: all 25 processes exited cleanly, four Hydra
  instances saved trajectories/timing data in about 1.1 seconds, launch exit
  code 0. No invalid-context abort or launch SIGTERM remained.

Synthetic RGB-D is deliberately simple. These results prove ROS wiring,
concurrency, namespace isolation and online graph updates, not forest
reconstruction quality.

## Run commands

```bash
make robotics-image
make build
make phase1-synthetic             # terminal 1
make accept-phase1                # terminal 2
make phase2-synthetic             # terminal 1
make accept-phase2                # terminal 2
```

Navigator override examples:

```bash
MNS_NAVIGATOR=coverage make phase1-synthetic
MNS_NAVIGATOR=nav2 make phase1-synthetic
make models
make robotics-ml-image
make phase1-synthetic-diffusion
```

Real Isaac staged entry points:

```bash
make gpu-preflight
make isaac-compatibility
make isaac-minimal
make models
make phase1
make phase2
```

## Known issues and external blockers

- NVIDIA's compatibility checker reports this host's 8.59 GB VRAM below its
  10 GB threshold and 16.49 GB RAM below 32 GB. Minimal PhysX and one 320x240
  camera work, but four-camera/Hydra/Diffusion load may be resource-limited.
- Articulated Go2 RL control, real Hydra artifacts, real Diffusion/Nav2, UAV
  aerial reconstruction and real 2 Go2 + 2 UAV concurrency are not yet
  validated.
- The complete initial coverage route is long. Sensor-depth ingestion is live
  and residual planning is unit-tested, but a full-duration online mission that
  naturally reaches and executes both residual passes has not been run.
- `PipelineStatus.dropped` estimates source timestamp gaps; pinned Hydra does
  not expose its internal queue-drop counter. Per-frame mapping latency remains
  a future measurement, as explicitly stated in the status detail.
- Automatic bag recording is not part of bringup. The checked-in MCAP policy is
  for an optional manual/sidecar recorder if needed.

## Architecture decisions

- Ubuntu 22.04 remains the host; ROS 2 Jazzy lives in an Ubuntu 24.04 container.
- Isaac Sim 5.1.0 / Isaac Lab 2.3.1 is the only simulator combination.
- Hydra-ROS is commit-pinned because there is no stable ROS 2 release matching
  this integration; its complete dependency graph is pinned too.
- An Ianvs 20 ms wall-sleep compatibility patch plus Hydra `force_shutdown`
  removes the Jazzy SIGINT context race in multi-instance launches.
- One Hydra instance per robot is the Phase 2 isolation mechanism. There is no
  global graph or implicit cross-robot state.
- Isaac's internal Jazzy ROS bridge is used inside simulation; the boundary is
  standard ROS messages, not shared Python modules or custom joint messages.
- Go2 locomotion is a motion backend, never part of Diffusion navigation.
- UAVs remain kinematic sensor platforms for this phase.

## ForestNavigation migration state

Migrated and reorganized: Go2 Isaac assets and actor inference, RGB-D history
and NavDiffusion inference, pure pursuit, zigzag/connected/sensor/residual
coverage, A* connectors, mapped-depth safety, stall recovery, Nav2 handoff,
kinematic UAV behavior and deterministic multi-team partitioning.

Intentionally excluded: ROS 1 handoff/conversion/validation, 1000 m reliability
and static reconstruction benchmarks, training, historical wrappers, duplicate
renderers/assets, backup/final scripts and conflicting environment freezes.
Exact attribution, changes and model hashes are in
`docs/forestnavigation_migration.md` and `THIRD_PARTY_NOTICES.md`.

## Remaining work after the blocker is removed

1. Validate the original Go2 locomotion actor against the articulated robot,
   including pose stability, forward/turn/stop response and command timeout.
2. Execute Phase 1 Coverage, Diffusion and Nav2 against the articulated Go2,
   checking joint stability, collisions, rates and concurrent Hydra artifacts.
3. Execute the 2 Go2 + 2 UAV roster, inspect each independent graph/mesh and
   specifically evaluate the downward UAV RGB-D/semantic reconstruction.
4. Run a full initial-plus-residual coverage mission and tune only configuration
   thresholds if the real sensor geometry requires it.
5. For real deployment, implement hardware-specific sensor calibration/time
   sync, Unitree high-level motion and UAV flight-stack adapters. Navigation and
   mapping interfaces should not change.

Future global DSG research would additionally need inter-robot transforms,
loop closure, fusion policy, entity reconciliation and a global graph owner.
Those modules are intentionally not implemented in Phase 1/2.
