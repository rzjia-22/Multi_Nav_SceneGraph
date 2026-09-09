# Development environment

## Runtime image strategy

The project has three images with one frozen dependency direction:

1. `multi-nav-scenegraph/robotics:jazzy-hydra-8f3b7e3` — Ubuntu 24.04,
   ROS 2 Jazzy, Nav2, project nodes and the complete pinned Hydra workspace.
2. `multi-nav-scenegraph/robotics-ml:jazzy-torch-2.7.1` — optional child of
   the robotics image containing fixed PyTorch/NavDiffusion dependencies.
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
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
docker compose --profile simulation build simulation
```

An NGC pull may require Isaac EULA acceptance. The simulation runtime requires
a working host NVIDIA driver, NVIDIA Container Toolkit, `libcuda.so.1` and a
Vulkan-capable device. ROS builds, tests, Hydra and synthetic acceptance do not
require a GPU.

On the current host, `nvidia-smi` cannot access a usable driver. A bounded
startup with the locally cached Isaac Lab 2.3.0 / Isaac Sim 5.1 image verified
AppLauncher arguments, Replicator extension loading, internal Jazzy imports and
project module imports, then stopped at `SimulationContext`/PhysX Fabric with
missing CUDA/Vulkan. That is an external host blocker, not an application
compile or ROS bridge failure. Re-run the official 2.3.1 GPU path after the
driver stack is repaired.

## DDS and throughput

The checked-in CycloneDDS profile avoids mandatory host sysctl changes. For
sustained four-camera runs, raising `net.core.rmem_max` and
`net.core.wmem_max` to at least 16 MiB is a recommended optimization, not a
startup requirement. Use the project acceptance observer for topic/TF/mapping
health and `docker stats --no-stream` for container CPU and memory usage.

## Runtime outputs and bags

Set `MNS_RUN_ID` to choose a reproducible output directory. Only letters,
digits, dots, underscores and hyphens are accepted. Hydra writes beneath
`runs/<run-id>/<robot>/hydra`; `runs/`, `bags/`, build products, caches and
large weights are ignored.

Optional ROS 2 recording uses MCAP and the topic list in
`config/system/recording.yaml`. It is never replayed, converted to ROS 1 or
required during normal online mapping.
