# Third-party notices

This project interoperates with ROS 2, Nav2, Hydra/Hydra-ROS, NVIDIA Isaac Sim,
Isaac Lab, and RSL-RL. Those projects remain under their respective licenses.

The robotics image builds the pinned MIT-SPARK source graph in
`upstream.repos`. `containers/robotics/ianvs-jazzy-shutdown.patch` carries a
small Jazzy shutdown compatibility change against Ianvs revision
`36f09956de88b44a579abe17bbfdcd0cc74eb929`; upstream copyright and license
headers remain intact in the patched source.

## ForestNavigation reference implementation

Repository: <https://github.com/LxRoboticsLab/ForestNavigation>

Audited revision: `0b29c399754f510499bfe9cc9d592cba215a7161`.

The upstream README identifies the project as BSD-3-Clause, although the
audited revision does not contain the referenced repository-root `LICENSE`
file. To avoid importing that ambiguity, this repository reimplements the ROS
node structure and most adapters. Files that retain algorithmic code derived
from ForestNavigation carry an explicit attribution header and are covered by
this project's BSD-3-Clause license. The RSL-RL copy bundled by the upstream
repository is not vendored here; the upstream package is consumed separately.

The following ideas and behavior were used as migration references:

- boustrophedon/zigzag and connected-free-space coverage planning;
- sensor-observed free-space accounting and residual-route planning;
- monotonic coverage progress tracking and pure-pursuit path following;
- mapped-obstacle safety overrides and command-response stall recovery;
- RGB-D history preparation, goal-conditioned diffusion inference, and the
  separation between trajectory generation and following;
- velocity-command adaptation for an RL locomotion policy;
- kinematic UAV zigzag collection and 2 Go2 + 2 UAV field partitioning;
- ROS 2 topic/frame naming lessons from the single- and multi-team collectors.

The project-owned NavDiffusion V0 network retains architectural attribution to
that audited revision but does not use its Lightning training wrapper. Its
EfficientNet-B0 initialization uses torchvision 0.22.1's official
`IMAGENET1K_V1` weights from PyTorch; the source URL and SHA256 are embedded in
the versioned training evidence and checkpoint metadata.

No ROS 1 handoff, static benchmark framework, historical experiment wrappers,
bundled RSL-RL source, generated forest assets, or large checkpoints are
copied into this repository.
