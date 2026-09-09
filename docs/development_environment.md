# Development environment

## Runtime image strategy

The project has three images with one frozen dependency direction:

1. `multi-nav-scenegraph/robotics:jazzy-hydra-8f3b7e3` — Ubuntu 24.04,
   ROS 2 Jazzy, Nav2, project nodes and the complete pinned Hydra workspace.
2. `multi-nav-scenegraph/robotics-ml:jazzy-torch-2.7.1` — optional child of
   the robotics image containing fixed PyTorch/NavDiffusion dependencies and
   explicit NVIDIA compute/utility passthrough for real GPU inference.
3. `multi-nav-scenegraph/simulation:isaac-lab-2.3.1` — official NGC Isaac Lab
   2.3.1 image with Isaac Sim 5.1.0.

Coverage and Nav2 do not pay the ML-image cost. The simulator does not source
the robotics image or project custom messages: its entrypoint exposes Isaac's
internal Jazzy `rclpy` and shared libraries from
`/isaac-sim/exts/isaacsim.ros2.bridge/jazzy`, then publishes standard ROS types.

All processes use host networking, Cyclone DDS, the same `ROS_DOMAIN_ID`
(default 42) and simulation time. This also supports later distribution across
a robot computer and companion computer without moving software boundaries.

## Version freeze

ROS 2 Jazzy is the sole application distribution. Hydra-ROS and every source
dependency are commit-pinned in `upstream.repos`; branches and `latest` tags are
not accepted. The only carried patch is
`containers/robotics/ianvs-jazzy-shutdown.patch`, which replaces context-bound
`rclcpp::WallRate::sleep()` in Ianvs spin loops with a 20 ms standard-library
wall sleep. It prevents a Jazzy SIGINT context race and does not alter Hydra
processing.

Isaac Lab 2.3.1 was selected over beta lines to retain Isaac Sim 5.1 stability
and a small migration distance from the audited Unitree/Crazyflie assets. Do
not introduce a second Isaac or ROS distribution into this repository.

## Build and test

```bash
make robotics-image
make build
make validate
docker compose config --quiet
```

The ML image requires the locally audited ForestNavigation checkout because
only its `navdiffusion` Python package is copied into the image:

```bash
make models
make robotics-ml-image
make test-models
```

`make models` uses Git LFS for only the two required files, verifies the remote
and revision, checks fixed SHA-256 digests and creates stable symlinks:

- `models/navdiffusion.ckpt`
- `models/go2_locomotion.pt`

The full `models/` directory is ignored. The ML Docker build additionally
verifies the source checkout revision and selected-source cleanliness before
copying it. PyTorch loaders use `weights_only=True` and strict tensor shapes.

## Host GPU checks

```bash
make gpu-preflight
make isaac-compatibility
make isaac-minimal
```

These are deliberately layered. `gpu-preflight` verifies host NVML, a CUDA
container, the registered NVIDIA Docker runtime and graphics capabilities.
`isaac-compatibility` runs NVIDIA's Vulkan/system checker. `isaac-minimal` then
starts the frozen 2.3.1 image, drops one rigid body for 120 GPU PhysX steps and
requires a machine-readable PASS marker. An NGC pull may require EULA
acceptance. ROS builds, tests, Hydra and synthetic acceptance do not require a
GPU.

The simulation service mounts this repository at `/mns`; `/workspace` belongs
to the official image and contains `/workspace/isaaclab`. Do not mount the
project over that directory. The current host produced `GPU_PREFLIGHT_PASS`,
enumerated Vulkan on an RTX 4060 Laptop GPU and passed the finite PhysX test on
Isaac Lab 2.3.1. The official checker reports 8.59 GB VRAM below 10 GB and 16.49
GB RAM below 32 GB, so those are measured resource risks rather than hidden as
software failures. Inside the Codex filesystem sandbox `/dev/nvidia*` is not
visible; GPU commands must execute through the host-authorized Docker path.

The simulation entrypoint sources Isaac's Python environment and then `exec`s
the runtime, so Python is container PID 1. Docker SIGINT therefore stops the
physics loop cleanly instead of killing a wrapper shell after its grace period.

For real message inspection, run the normal simulation process and then:

```bash
MNS_ACCEPTANCE_DURATION=30 make accept-isaac-sensors
```

This observer decodes image payloads and checks encodings, metric depth,
integer semantic IDs, intrinsics, timestamps, rates and TF at the image stamp.
The normal runtime loads `config/simulation/forest.yaml`; a finite diagnostic
run can add `--max-steps N` to the simulation entrypoint and must print
`MNS_ISAAC_RUNTIME_RESULT` before exiting with code zero.

The general Phase 1/2 observer reports both simulation-time frequency and wall
throughput. A low wall rate is a resource warning when message stamps still
show the configured sensor rate; it is not mislabeled as a ROS contract failure.

The articulated Go2 motion gate uses an isolated southern lane, independently
of navigation and Hydra. Start the simulator in one terminal:

```bash
docker compose --profile simulation run --rm --name mns-go2-motion simulation \
  /mns/containers/simulation/entrypoint.sh \
  --scenario phase1_go2 \
  --system-config /mns/config/robots/phase1_motion_acceptance.yaml \
  --go2-backend rl --headless --enable_cameras
```

Then exercise the public ROS command boundary from another terminal:

```bash
make accept-go2-motion
```

The observer requires an upright base through stand, forward, moving turn,
explicit stop, command-timeout and post-command settling phases. A failure is a
motion-backend blocker; it must not be replaced with kinematic evidence or
hidden by relaxing the fall thresholds. Remove `--enable_cameras` only for an
explicit sensorless isolation run, never for final concurrency evidence.

`make probe-go2-upstream` is a separate finite diagnostic. It runs the frozen
actor in ForestNavigation's original Gym task, including the same navigation
command profile, to distinguish a corrupt checkpoint or migrated tensor
contract from behavior that fails only at the project's runtime boundary.

## DDS and throughput

The checked-in CycloneDDS profile avoids mandatory host sysctl changes. For
sustained four-camera runs, raising `net.core.rmem_max` and
`net.core.wmem_max` to at least 16 MiB is a recommended optimization, not a
startup requirement. Use the project acceptance observer for topic/TF/mapping
health and `docker stats --no-stream` for container CPU and memory usage.
The measured four-camera/four-Hydra run on the current host maintained 10 Hz in
simulation time but only about 2.4 Hz wall throughput. It used about 4.5 GB
VRAM, 3.0 GB Isaac RAM and 6.4 GB robotics RAM; CPU/Hydra concurrency is the
dominant practical constraint.

## Runtime outputs and bags

Set `MNS_RUN_ID` to choose a reproducible output directory. Only letters,
digits, dots, underscores and hyphens are accepted. Hydra writes beneath
`runs/<run-id>/<robot>/hydra`; `runs/`, `bags/`, build products, caches and
large weights are ignored.

Optional ROS 2 recording uses MCAP and the topic list in
`config/system/recording.yaml`. It is never replayed, converted to ROS 1 or
required during normal online mapping.
