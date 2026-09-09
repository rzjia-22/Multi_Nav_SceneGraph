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

## Package responsibilities

`mns_core` owns validated robot descriptions, topic/frame naming and pure data
contracts. It has no ROS imports.

`mns_navigation` owns the replaceable strategies:

- connected or zigzag coverage route generation with an obstacle-aware A*
  connector and residual-cell hooks;
- monotonic path progress and pure-pursuit following;
- original goal-conditioned NavDiffusion inference from five RGB-D frames,
  followed by the same high-level trajectory follower;
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
and previous action. The frozen actor produces 12 normalized offsets, which
are scaled and added to the default joint pose. Navigation never sees those
joint targets.

Replicator produces RGB, metric depth and Isaac ground-truth semantic labels.
Semantic metadata are remapped to the checked-in eight-class Hydra label space.
Each camera is calibrated with `CameraInfo`, stamped with simulation time and
connected to its robot base by prefixed TF.

UAV motion is intentionally kinematic in Phase 2: `cmd_vel_safe` updates the
pose while the platform publishes the same odom, TF and camera contract as
Go2. Full flight dynamics are outside the current scope.

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
