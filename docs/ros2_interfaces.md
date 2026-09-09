# ROS 2 interfaces

All names below are relative to a robot namespace `/ROBOT_ID` unless shown as
absolute. Namespace isolation and TF frame prefixes are both required because
TF frame IDs are not scoped by ROS namespaces.

## Sensor contract

| Topic | Type | Required | Contract |
| --- | --- | --- | --- |
| `camera/color/image_raw` | `sensor_msgs/Image` | yes | `rgb8`, registered camera |
| `camera/color/camera_info` | `sensor_msgs/CameraInfo` | yes | calibrated and stamped in optical frame |
| `camera/depth/image_rect` | `sensor_msgs/Image` | yes | `32FC1` metres; `16UC1` mm is adapter-compatible |
| `camera/semantic/image_raw` | `sensor_msgs/Image` | yes | one-channel integer class IDs, not RGB colors |
| `odom` | `nav_msgs/Odometry` | yes | parent `ROBOT_ID/odom`, child `ROBOT_ID/base_link` |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | yes | pose resolvable at each image timestamp |
| `/clock` | `rosgraph_msgs/Clock` | simulation | one shared publisher, all nodes use simulation time |

The four camera streams use reliable, volatile QoS because the pinned
Hydra-ROS input requests reliable subscriptions, including its initial
CameraInfo lookup. Mission goals use reliable, transient-local QoS so a
navigator launched shortly after the coordinator still receives its goal.

The Isaac adapter maps Replicator `idToLabels` metadata to this stable label
space: unknown 0, ground 1, tree trunk 2, foliage 3, rock 4, building 5,
robot 6 and other object 7. Replacing Isaac GT semantics with a real inference
source does not change the topic contract.

## Commands and missions

| Topic/action | Type | Producer → consumer |
| --- | --- | --- |
| `mission/goal` | `geometry_msgs/PoseStamped` | coordinator/operator → selected navigator |
| `mission/path` | `nav_msgs/Path` | coverage or Diffusion → observer/follower |
| `navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | Nav2 adapter → Nav2 |
| `mission/status` | `mns_interfaces/MissionStatus` | selected navigator → coordinator/observer |
| `cmd_vel/navigation` | `geometry_msgs/Twist` | selected navigator → command arbiter |
| `cmd_vel/safety` | `geometry_msgs/Twist` | safety/recovery → command arbiter |
| `cmd_vel_safe` | `geometry_msgs/Twist` | command arbiter → robot motion backend |

Only one navigator is instantiated per robot. The arbiter chooses a fresh
safety command before a fresh navigation command and emits zero when both are
stale. Go2 joint targets are simulator-local implementation details and are
never a navigation or network interface. The Isaac Go2 adapter independently
zeros a stale `cmd_vel_safe` after 0.3 simulation seconds, so losing the ROS
publisher cannot leave a latched locomotion command active.

`MissionStatus` contains `robot_id`, `navigator`, lifecycle `state`, progress
and a human-readable detail. Coverage reports route progress, Diffusion reports
RGB-D history readiness, and Nav2 reflects action acceptance/completion.

## Mapping outputs and health

| Topic | Type | Notes |
| --- | --- | --- |
| `hydra/frontend/dsg` | `hydra_msgs/DsgUpdate` | live frontend graph owned by pinned Hydra |
| `hydra/backend/dsg` | `hydra_msgs/DsgUpdate` | optimized independent per-robot graph |
| `hydra/status` | `std_msgs/String` | Hydra-owned textual runtime status |
| `hydra/pipeline_status` | `mns_interfaces/PipelineStatus` | project-owned input/output health stream |

`PipelineStatus` reports state, RGB input rate, backend DSG publication rate,
received RGB count, estimated missing source frames and a latency placeholder.
`dropped` is derived from gaps in monotonically stamped RGB input at the
configured expected rate; it is not an internal Hydra queue counter. Likewise,
DSG publication rate is throughput and `latency_ms` is not yet a per-frame
mapping completion measurement. The wording is intentional so later detailed
latency instrumentation can be added without changing the message type.

Hydra writes graph, mesh, trajectory and timing artifacts below
`runs/<run-id>/<robot>/hydra`. Each process receives a distinct numeric
`robot_id`, prefixed map/odom/base frames and output directory. The Jazzy
shutdown configuration drains no queued sensor backlog (`force_shutdown`) but
does save finalized Hydra outputs before exiting.

## TF policy

There are no unprefixed robot frames. The robot adapter publishes dynamic
`ROBOT_ID/odom → ROBOT_ID/base_link` and static
`ROBOT_ID/base_link → ROBOT_ID/camera_link → ROBOT_ID/camera_optical_frame`.
Hydra may publish `ROBOT_ID/map → ROBOT_ID/odom`. The runtime acceptance tool
checks every required prefixed frame for every robot.

## Optional recording

`config/system/recording.yaml` defines MCAP topics for reproducible debugging.
Recording is an optional subscriber side channel; no bag, replay or ROS 1
conversion is present in the live Hydra path.
