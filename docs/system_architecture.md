# System architecture

## Runtime boundaries

```text
Isaac / synthetic / future real adapter
  ├── RGB + depth + semantics + CameraInfo ─► one Hydra pipeline per robot ─► DSG
  ├── odom + TF + clock ───────────────────► navigation and mapping
  └── motion backend ◄── cmd_vel_safe ◄── command arbiter ◄── navigator
                                                        ▲
                         coverage path + follower ──────┤
                         diffusion path + follower ─────┤
                         Nav2 planning + control ────────┘
```

Mission planning decides where to go; navigation decides how to get there.
All three navigation modes emit `geometry_msgs/Twist`. Only a motion backend
may translate that command into joint targets or kinematic motion. Mapping and
navigation run concurrently and neither imports Isaac APIs.

Container boundaries are deployment choices, not software boundaries. The
robotics image contains ROS 2 Jazzy, Hydra and project nodes. The simulation
image contains Isaac Sim/Lab and uses Isaac's internal Jazzy ROS bridge to
exchange standard DDS messages over host networking.

Dataset V0 is a separate offline research-data consumer of Isaac:

```text
deterministic Research Forest + privileged map
  ├── A* + LOS smoothing ─► Pure Pursuit ─► kinematic surrogate
  └── moving D435i-like rig ─► RGB + raw/registered depth + IMU ─► HDF5
```

The Research Forest specification and full 70-episode collection are validated
and human-accepted. It is rendered only by the shared Isaac scene
builder: official TerrainImporter/TerrainGenerator APIs provide terrain,
NVIDIA USD references provide visible vegetation, MDL provides ground and an
HDR dome plus sun provide lighting.

It reuses the simulation image but is not a ROS package and does not make
Hydra a collection dependency. `research_data/` owns deterministic scene,
expert, schema, validation and visualization logic; the Isaac camera/surrogate
runtime stays in `mns_simulation`. The original integration forest remains
unchanged while `research_scenes/dataset_v0/` stores research realizations.

## Package responsibilities

`mns_core` owns validated robot descriptions, topic/frame naming and pure data
contracts. It has no ROS imports.

`mns_navigation` owns the replaceable strategies:

- connected or zigzag coverage route generation with an obstacle-aware A*
  connector and residual-cell hooks;
- monotonic path progress and pure-pursuit following;
- legacy goal-conditioned NavDiffusion inference plus project-owned
  NavDiffusion V0 training/inference from five RGB-D frames, selected by
  configuration and followed by the same high-level trajectory follower;
- a namespaced Nav2 lifecycle/action adapter;
- mapped-depth safety override and bounded stall recovery logic.

`mns_motion` arbitrates navigation and safety commands using freshness and a
dead-man timeout, publishing only `cmd_vel_safe`. A future Unitree hardware
adapter subscribes to this same boundary.

`mns_mapping` creates one namespaced Hydra process per enabled robot, remaps
the formal sensor contract and publishes a separate health stream. It does not
modify mapping algorithms or share a global graph.

`mns_multi_robot` expands a robot roster and publishes configured mission goals
with transient-local QoS. It tracks mission lifecycle, but does not implement
global DSG fusion, relative localization or complex allocation.

`mns_simulation` is the only Isaac-dependent package. One process owns the
shared scene, clock and all robots. Phase 1 creates one Go2; Phase 2 creates two
Go2 articulations and two Crazyflie-derived kinematic UAV sensor platforms.
The synthetic node implements the same ROS contract solely for CPU integration
testing.

The lightweight acceptance forest is defined once in
`config/simulation/forest.yaml`. The Isaac adapter creates collision-enabled
trunks plus semantic foliage from it, while known-area connected coverage reads
the same tree centres as inflated obstacles. This keeps the baseline route and
the rendered world consistent without turning coverage into exploration or
coupling navigation code to Isaac APIs.

## Isaac control and sensing

The planned runtime rates are 200 Hz physics, 50 Hz Go2 actor inference and
10 Hz sensor publication. The Go2 adapter builds the audited 48-value velocity
observation from body velocity, gravity projection, Twist command, joint state
and previous action. The frozen actor produces 12 policy-space offsets, which
are scaled and added to the default joint pose. Navigation never sees those
joint targets. The adapter asserts the checkpoint's exact joint order and
default pose, starts the actor at 50 Hz from the first physics frame, publishes
full roll/pitch/yaw odometry, and applies a simulation-time command deadman.
Rendering remains 10 Hz but uses a one-physics-tick rendering timestep so an
RTX update never bypasses policy ticks. Stand, forward, moving turn, stop and
deadman behavior pass with real camera rendering enabled.

Isaac Lab's managed Camera sensor produces RGB, metric depth and Isaac
ground-truth semantic labels.
Semantic metadata are remapped to the checked-in eight-class Hydra label space.
Each camera is calibrated with `CameraInfo`, stamped with simulation time and
connected to its robot base by prefixed TF.
For multi-robot operation, one batched Camera sensor is created per robot kind
using an Isaac prim-path expression, then each indexed output is published by
its own ROS node. This avoids duplicate children on instanceable USD clones
without sharing ROS namespaces or mapping state.

UAV motion is intentionally kinematic in Phase 2: `cmd_vel_safe` updates the
pose while the platform publishes the same odom, TF and camera contract as
Go2. Its nadir camera is attached to Crazyflie's moving `body` link. The current
planar depth-safety node is Go2-only because a downward mapping image is not a
valid forward collision measurement. Full flight dynamics and UAV-specific
collision sensing are outside the current scope.

## Multi-robot isolation

For `go2_1`, topics live below `/go2_1`, while frame IDs are globally unique:
`go2_1/map`, `go2_1/odom`, `go2_1/base_link`, `go2_1/camera_link` and
`go2_1/camera_optical_frame`. Hydra publishes the independent graph on
`/go2_1/hydra/backend/dsg` and writes only to that robot's run directory.

The same launch generator reads `config/robots/phase1.yaml` or
`config/robots/phase2.yaml`; there are no separate single- and multi-robot
implementations. Repository validation rejects duplicate robot IDs, duplicate
Hydra IDs and TF frame collisions.

## Deferred extension points

- `MotionBackend`: a real Unitree/Jetson or UAV flight-stack adapter.
- `SemanticSource`: a real-camera inference source replacing Isaac GT labels.
- `MappingBackend`: another online mapper consuming the same sensor contract.
- `SceneGraphConsumer`: a future fusion/reconciliation input.

Global DSG fusion, inter-robot loop closure, relative localization and entity
reconciliation are intentionally absent.
