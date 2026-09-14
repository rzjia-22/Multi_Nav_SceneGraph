# ForestNavigation migration notes

The complete upstream tree was inventoried and relevant implementation files
were read at revision `0b29c399754f510499bfe9cc9d592cba215a7161`.
Responsibilities were moved into stable packages instead of copying the old
`scripts/` hierarchy or experiment wrappers.

## Migrated behavior

| Upstream responsibility | New home | Material change |
| --- | --- | --- |
| Go2 Isaac asset and velocity actor | `mns_simulation.isaac_runtime`, `go2_policy` | shared Isaac world; strict safe loader; 48-value observation to 12 joint offsets; Twist boundary |
| RGB-D history and goal-conditioned diffusion | `mns_navigation.diffusion`, `diffusion_node`, `navdiffusion_v0` | legacy package retained for checkpoint compatibility; project-owned current-PyTorch model/trainer shares preprocessing with the selectable ROS predictor |
| Pure Pursuit | `mns_navigation.path_follower` | ROS-independent implementation shared by route-producing navigators |
| Zigzag coverage | `mns_navigation.coverage` | immutable `CoveragePath` and stable route API |
| Connected Coverage and A* connector | `mns_navigation.coverage` | obstacle-inflated reachable component; no Isaac imports or hard-coded scene path |
| Sensor Coverage | `mns_navigation.sensor_coverage` | timestamped depth rays transformed by ROS TF into a reachable free-space grid |
| Residual Coverage | `mns_navigation.sensor_coverage`, `coverage_node` | connected residual components become bounded A* route passes after initial route completion |
| Mapped obstacle avoidance | `mns_navigation.safety` | separate high-priority Twist source behind the arbiter |
| Stall recovery | `mns_navigation.safety` | bounded command-response state machine, independent of locomotion |
| Nav2 configuration and goal handoff | `config/nav2`, `nav2_adapter` | Jazzy lifecycle/action API, fully namespaced frames and commands |
| Single UAV collector | `mns_simulation` | kinematic aerial sensor abstraction with live RGB-D/semantic ROS topics; no bag prerequisite |
| Go2 + UAV and two-team runners | `mns_bringup`, `mns_multi_robot` | YAML roster and one shared simulation instead of experiment-specific entry scripts |
| Two-team partition | `mns_multi_robot.partition` | deterministic general quadrant primitive used by configuration, not a coupled runner |
| Forest generator design reference | `research_data.forest`, `mns_simulation.research_forest_scene`, `config/research_forests` | replaced custom sine/primitive visuals with seeded Isaac Lab terrain, current Isaac 5.1 NVIDIA Blue Berry Elder/Gray Birch USDs, MDL ground, HDR sky and hidden conservative colliders; no old script loop copied |

The source files that retain adapted coverage/partition behavior include the
audited revision in their module header. The exact upstream `navdiffusion`
Python package is copied only from a locally verified checkout into the ML
image because checkpoint compatibility depends on that model definition.
No source from `scripts/forest_generator/forest_generator_isaaclab.py` or
`scripts/forest_generator/test.py` is copied into the Dataset V0 generator.

For the accepted Research Forest visual domain, the old `Collected_forest_v2` and
`Collected_forest_v5` mapping records were also audited. They identify Blue
Berry Elder, Natural/Dirt MDL and their bark/leaf/ground textures; v5 additionally
contains Holly and an unrelated farmhouse. The new registry resolves assets
through the versioned Isaac Sim 5.1 cloud root, adds Gray Birch from that same
official tree directory, and intentionally excludes Holly (a shrub) and the
farmhouse. External NVIDIA assets remain URI references and are not copied from
the upstream LFS collection into this repository.

## Checkpoint handling

The two required Git LFS assets are fetched selectively and remain outside
version control:

| Stable project path | Upstream file | SHA-256 |
| --- | --- | --- |
| `models/navdiffusion.ckpt` | `forest_nav/results/epoch=252-step=12903.ckpt` | `023d7b637a3ba2bea6ad918bb45add055ccd57581d09ccf4ed1f1ce56280dfde` |
| `models/go2_locomotion.pt` | `logs/rsl_rl/unitree_go2_rough/2025-11-03_01-41-16/model_5998.pt` | `f06f93d6606c4cb64a03e2a137db398d52c75b8467268a28cf97bac4faa7086f` |

Both files have been acquired and verified in the current workspace. The
Diffusion model loads its strict state dictionary with `weights_only=True` and
produces eight 2-D waypoints. The Go2 adapter reconstructs the actor-only MLP
from tensor keys, validates the `48 → 512 → 256 → 128 → 12` layout and never
loads arbitrary checkpoint Python objects. No retraining or conversion occurs.

The statement above applies to the two original upstream checkpoints. The
separate project-owned `mns_navdiffusion_v0` model was trained from the frozen
Dataset V0 using torchvision ImageNet weights and plain PyTorch 2.7.1. It does
not modify the pinned upstream checkout, restore Lightning 1.8/WandB, or copy
the upstream monolithic HDF5 workflow. Its architecture attribution and exact
training evidence are documented in `docs/navdiffusion_v0.md`.

The runtime additionally asserts the upstream joint ordering and default pose.
A 60-second probe against the original Gym environment and the project's ROS
motion gate both pass the same low-speed stand/forward/moving-turn/stop profile.
The first project failures were traced to an adapter scheduling error: the
rendering timestep was 100 ms inside a 5 ms outer loop, so an RTX update skipped
policy ticks. The migrated runtime now renders only on sensor frames while each
render advances one physics tick. No actor clipping or checkpoint replacement
was introduced.

## Intentionally excluded

The following are intentionally absent: ROS 1 handoff and bag conversion, ROS
1 validation, 1000 m reliability runs, static reconstruction benchmarks,
upstream training entry points, bundled RSL-RL source, duplicate generated forest
assets, renderer/review tools, backup/final scripts, legacy experiment wrappers
and conflicting environment freeze files.

No multi-robot global DSG, Hydra-Multi, inter-robot loop closure, relative
localization, fusion or entity reconciliation code was migrated.

## Licensing

The upstream README identifies ForestNavigation as BSD-3-Clause, but the
audited revision lacks its referenced repository-root license file. The
project therefore keeps explicit attribution and revision records, copies only
the checkpoint-compatible model package into an optional image, and does not
vendor the upstream RSL-RL tree. See `THIRD_PARTY_NOTICES.md`.
