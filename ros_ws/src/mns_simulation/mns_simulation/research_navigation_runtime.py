"""Isaac closed-loop runtime for NavDiffusion V0 on the DIABLO surrogate.

Isaac owns only the accepted Research Forest, D435i-like rendering and
kinematic robot adapter.  Navigation remains in the robotics-ML container and
crosses this process boundary exclusively through standard ROS 2 messages.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from isaaclab.app import AppLauncher


PROCESS_START = time.monotonic()
CHECKPOINT_SHA256 = "7b3bca67af26c2d839555e9b27a7a420762b572fa88664a227986174d9a68ae0"
ROBOT_ID = "diablo_1"


def parse_args():
    project_root = Path(os.environ.get("MNS_PROJECT_ROOT", "/mns"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=project_root / "config/research_forests/assets.yaml")
    parser.add_argument("--episodes", required=True, help="comma-separated validation episode IDs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-group", required=True)
    parser.add_argument("--run-root", type=Path, default=project_root / "runs/navdiffusion_v0_closed_loop")
    parser.add_argument("--artifact-root", type=Path, default=project_root / "artifacts/navdiffusion_v0_closed_loop")
    parser.add_argument("--physics-rate", type=float, default=100.0)
    parser.add_argument("--maximum-duration", type=float, default=60.0)
    parser.add_argument("--command-timeout", type=float, default=0.30)
    parser.add_argument("--connection-timeout", type=float, default=45.0)
    parser.add_argument("--capture-review", action="store_true")
    parser.add_argument("--fail-on-episode-failure", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app
APP_READY = time.monotonic()


def _percentile(values, percentile: float) -> float | None:
    import numpy as np

    return float(np.percentile(values, percentile)) if len(values) else None


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.inprogress")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _command(message) -> list[float]:
    return [float(message.linear.x), float(message.linear.y), float(message.angular.z)]


class RosEndpoint:
    """State shared by ROS callbacks and the deterministic Isaac loop."""

    def __init__(self, node, publisher) -> None:
        from geometry_msgs.msg import PoseStamped, Twist
        from mns_interfaces.msg import MissionStatus
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Empty, String

        self.node = node
        self.publisher = publisher
        self.sim_time = 0.0
        self.navigation_command = [0.0, 0.0, 0.0]
        self.safe_command = [0.0, 0.0, 0.0]
        self.navigation_stamp = float("-inf")
        self.safe_stamp = float("-inf")
        self.status = "starting"
        self.status_detail = ""
        self.planning_diagnostics: list[dict] = []
        self.reset_pub = node.create_publisher(Empty, "mission/reset", 10)
        self.episode_pub = node.create_publisher(String, "mission/episode_id", 10)
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_pub = node.create_publisher(PoseStamped, "mission/goal", goal_qos)
        self.subscriptions = [
            node.create_subscription(Twist, "cmd_vel/navigation", self._navigation, 20),
            node.create_subscription(Twist, "cmd_vel_safe", self._safe, 20),
            node.create_subscription(MissionStatus, "mission/status", self._status, 10),
            node.create_subscription(String, "mission/planning_diagnostics", self._diagnostic, 20),
        ]

    def _navigation(self, message) -> None:
        self.navigation_command = _command(message)
        self.navigation_stamp = self.sim_time

    def _safe(self, message) -> None:
        self.safe_command = _command(message)
        self.safe_stamp = self.sim_time

    def _status(self, message) -> None:
        self.status = message.state
        self.status_detail = message.detail

    def _diagnostic(self, message) -> None:
        try:
            value = json.loads(message.data)
        except (TypeError, ValueError):
            value = {"status": "FAIL", "failure_class": "MODEL", "error": "invalid diagnostic JSON"}
        value["received_sim_time_s"] = self.sim_time
        self.planning_diagnostics.append(value)

    def clear_episode(self) -> None:
        self.navigation_command = [0.0, 0.0, 0.0]
        self.safe_command = [0.0, 0.0, 0.0]
        self.navigation_stamp = float("-inf")
        self.safe_stamp = float("-inf")
        self.status = "resetting"
        self.status_detail = ""
        self.planning_diagnostics.clear()

    def publish_reset(self) -> None:
        from std_msgs.msg import Empty

        self.reset_pub.publish(Empty())

    def publish_episode(self, episode_id: str) -> None:
        from std_msgs.msg import String

        message = String()
        message.data = episode_id
        self.episode_pub.publish(message)

    def publish_goal(self, goal_xyz, frame_id: str) -> None:
        from geometry_msgs.msg import PoseStamped
        from mns_simulation.ros_publisher import time_message

        message = PoseStamped()
        message.header.stamp = time_message(self.sim_time)
        message.header.frame_id = frame_id
        message.pose.position.x = float(goal_xyz[0])
        message.pose.position.y = float(goal_xyz[1])
        message.pose.position.z = float(goal_xyz[2])
        message.pose.orientation.w = 1.0
        self.goal_pub.publish(message)

    def bounded_safe_command(self, now: float, robot: dict, timeout: float) -> tuple[float, float]:
        import numpy as np

        if now - self.safe_stamp > timeout:
            return 0.0, 0.0
        linear = float(np.clip(
            self.safe_command[0],
            -float(robot["motion"]["max_reverse_speed_mps"]),
            float(robot["motion"]["max_forward_speed_mps"]),
        ))
        angular = float(np.clip(
            self.safe_command[2],
            -float(robot["motion"]["max_yaw_rate_radps"]),
            float(robot["motion"]["max_yaw_rate_radps"]),
        ))
        return linear, angular


def _minimum_clearance(scene: dict, x: float, y: float, robot_radius: float):
    values = []
    for tree in scene["trees"]:
        tx, ty = (float(value) for value in tree["position_m"])
        tree_radius = float(tree["collision_proxy"]["radius_m"])
        values.append((math.hypot(x - tx, y - ty) - tree_radius - robot_radius, tree["tree_id"]))
    return min(values, key=lambda item: item[0])


def _trajectory_clearance(scene: dict, points, robot_radius: float) -> float | None:
    if not points:
        return None
    return min(
        _minimum_clearance(scene, float(point[0]), float(point[1]), robot_radius)[0]
        for point in points
    )


def _classify_failure(reason: str, diagnostics: list[dict], override_fraction: float) -> str | None:
    if not reason:
        return None
    if reason.startswith("sensor") or reason == "history_timeout":
        return "SENSOR"
    if reason.startswith("model") or any(item.get("status") == "FAIL" for item in diagnostics):
        return "MODEL"
    if reason == "collision":
        if diagnostics and diagnostics[-1].get("control_minimum_clearance_m", 1.0) <= 0.0:
            return "MODEL"
        return "CONTROLLER"
    if override_fraction > 0.50:
        return "SAFETY"
    if reason == "stall" and len(diagnostics) >= 3:
        return "COVARIATE_SHIFT"
    if reason in {"timeout", "stall"}:
        return "CONTROLLER"
    return "SIMULATION"


def main() -> int:
    import numpy as np
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.ros2.bridge")
    import omni.usd
    import rclpy
    import torch
    from pxr import Gf, Semantics, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationContext

    from research_data.common import ROOT, camera_intrinsics, load_yaml
    from research_data.depth import align_depth_to_rgb, quantize_depth_z16
    from research_data.expert import validate_plan
    from mns_simulation.research_forest_scene import build_research_forest
    from mns_simulation.research_robot import (
        episode_mount_variation,
        make_camera,
        research_rig_pose,
        surface_frame,
    )
    from mns_simulation.ros_publisher import ResearchRobotPublisher

    if min(ARGS.physics_rate, ARGS.maximum_duration, ARGS.command_timeout) <= 0.0:
        raise ValueError("runtime rates and timeouts must be positive")
    episode_ids = [value.strip() for value in ARGS.episodes.split(",") if value.strip()]
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episodes must be a non-empty unique comma-separated list")
    if any(not value.startswith("validation_scene_") for value in episode_ids):
        raise ValueError("closed-loop runtime accepts validation episodes only")

    scene = load_yaml(ARGS.scene)
    registry = load_yaml(ARGS.assets)
    robot = load_yaml(ROOT / "config/robots/diablo_standing.yaml")
    sensor = load_yaml(ROOT / "config/sensors/d435i_navigation_v0.yaml")
    if scene["split"] != "validation":
        raise ValueError("closed-loop scene must belong to the validation split")
    if any(not value.startswith(scene["scene_id"] + "_episode_") for value in episode_ids):
        raise ValueError("every episode must belong to the selected validation scene")
    plans = {}
    for episode_id in episode_ids:
        path = ROOT / "datasets/dataset_v0/validation" / scene["scene_id"] / episode_id / "episode_plan.yaml"
        plan = load_yaml(path)
        if plan["split"] != "validation" or plan["scene_id"] != scene["scene_id"]:
            raise ValueError(f"invalid validation provenance for {episode_id}")
        validate_plan(scene, plan)
        plans[episode_id] = plan

    physics_dt = 1.0 / float(ARGS.physics_rate)
    sensor_rate = float(sensor["application_profile"]["logging_rate_hz"])
    state_rate = float(sensor["application_profile"]["state_logging_rate_hz"])
    sensor_interval = round(ARGS.physics_rate / sensor_rate)
    state_interval = round(ARGS.physics_rate / state_rate)
    if not math.isclose(sensor_interval * sensor_rate, ARGS.physics_rate):
        raise ValueError("D435i logging rate must divide physics rate")
    if not math.isclose(state_interval * state_rate, ARGS.physics_rate):
        raise ValueError("state rate must divide physics rate")

    simulation = SimulationContext(sim_utils.SimulationCfg(
        dt=physics_dt, render_interval=1, device=ARGS.device
    ))
    simulation.set_camera_view([13.0, 13.0, 12.0], [0.0, 0.0, 0.0])
    scene_started = time.monotonic()
    built = build_research_forest(simulation, scene, registry)
    scene_load_time = time.monotonic() - scene_started
    stage = omni.usd.get_context().get_stage()

    surrogate = UsdGeom.Xform.Define(stage, "/World/DiabloSurrogate")
    body_translate = surrogate.AddTranslateOp()
    body_orient = surrogate.AddOrientOp()
    body = UsdGeom.Cube.Define(stage, "/World/DiabloSurrogate/Body")
    body.CreateSizeAttr(1.0)
    body.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.10, 0.13)])
    body.AddScaleOp().Set(Gf.Vec3f(
        float(robot["surrogate"]["footprint"]["length_m"]),
        float(robot["surrogate"]["footprint"]["width_m"]),
        float(robot["surrogate"]["body_height_m"]),
    ))
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(robot["surrogate"]["body_height_m"]) / 2.0))
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    semantic = Semantics.SemanticsAPI.Apply(body.GetPrim(), "Semantics")
    semantic.CreateSemanticTypeAttr().Set("class")
    semantic.CreateSemanticDataAttr().Set("robot")

    rgb_cfg = sensor["application_profile"]["rgb"]
    depth_cfg = sensor["application_profile"]["depth_raw"]
    rgb_camera = make_camera(
        sim_utils, Camera, CameraCfg, "/World/RgbCamera", rgb_cfg, sensor_rate, ["rgb"]
    )
    depth_camera = make_camera(
        sim_utils, Camera, CameraCfg, "/World/DepthCamera", depth_cfg, sensor_rate,
        ["distance_to_image_plane"],
    )
    UsdGeom.Camera(stage.GetPrimAtPath("/World/RgbCamera")).GetExposureAttr().Set(
        float(rgb_cfg["simulated_exposure_compensation_stops"])
    )
    simulation.reset()

    configured_rgb = np.asarray(camera_intrinsics(
        *rgb_cfg["resolution"], rgb_cfg["fov_deg"]["horizontal"], rgb_cfg["fov_deg"]["vertical"]
    ), dtype=np.float64)
    configured_depth = np.asarray(camera_intrinsics(
        *depth_cfg["resolution"], depth_cfg["fov_deg"]["horizontal"], depth_cfg["fov_deg"]["vertical"]
    ), dtype=np.float64)
    configured_rgb[1, 1], configured_depth[1, 1] = configured_rgb[0, 0], configured_depth[0, 0]
    rgb_intrinsics = rgb_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    depth_intrinsics = depth_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(rgb_intrinsics, configured_rgb, rtol=2.0e-3, atol=0.6):
        raise RuntimeError("Isaac RGB intrinsics differ from d435i_navigation_v0")
    if not np.allclose(depth_intrinsics, configured_depth, rtol=2.0e-3, atol=0.6):
        raise RuntimeError("Isaac depth intrinsics differ from d435i_navigation_v0")

    rclpy.init(args=None)
    node = rclpy.create_node("diablo_1_research_simulator", namespace=ROBOT_ID)
    publisher = ResearchRobotPublisher(node, ROBOT_ID, rgb_intrinsics, rgb_cfg["resolution"])
    endpoint = RosEndpoint(node, publisher)
    stop_requested = False
    exit_reason = "completed"

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested, exit_reason
        stop_requested = True
        exit_reason = signal.Signals(signum).name.lower()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    global_frame = 0

    def place_rig(x: float, y: float, yaw: float, sim_time: float, mount_variation):
        rig = research_rig_pose(
            built.surface, x, y, yaw, sim_time, robot, sensor, mount_variation
        )
        body_translate.Set(Gf.Vec3d(x, y, rig.ground_z))
        body_orient.Set(Gf.Quatf(
            float(rig.body_xyzw[3]), Gf.Vec3f(*[float(value) for value in rig.body_xyzw[:3]])
        ))
        orientation = torch.as_tensor(
            np.stack([rig.optical_wxyz]), dtype=torch.float32, device=simulation.device
        )
        rgb_camera.set_world_poses(
            torch.as_tensor(np.stack([rig.rgb_position]), dtype=torch.float32, device=simulation.device),
            orientation, convention="ros",
        )
        depth_camera.set_world_poses(
            torch.as_tensor(np.stack([rig.depth_position]), dtype=torch.float32, device=simulation.device),
            orientation, convention="ros",
        )
        return rig

    first_plan = plans[episode_ids[0]]
    x, y, _, yaw = (float(value) for value in first_plan["start_pose_xyzyaw"])
    first_mount = episode_mount_variation(robot, int(scene["scene_seed"]), episode_ids[0])
    rig = place_rig(x, y, yaw, 0.0, first_mount)
    ready_started = time.monotonic()
    ready = False
    while APP.is_running() and not stop_requested and time.monotonic() - ready_started < ARGS.connection_timeout:
        endpoint.sim_time = global_frame * physics_dt
        rclpy.spin_once(node, timeout_sec=0.0)
        render = (global_frame + 1) % sensor_interval == 0
        simulation.step(render=render)
        rgb_camera.update(physics_dt)
        depth_camera.update(physics_dt)
        global_frame += 1
        endpoint.sim_time = global_frame * physics_dt
        publisher.publish_clock(endpoint.sim_time)
        if global_frame % state_interval == 0:
            publisher.publish_pose(
                endpoint.sim_time, [x, y, rig.ground_z], rig.body_xyzw,
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                rig.body_to_optical_translation, rig.body_to_optical_xyzw,
            )
        if render:
            rgb = rgb_camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
            raw = depth_camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy().squeeze().astype(np.float32)
            z16, _ = quantize_depth_z16(raw, float(depth_cfg["depth_scale_m"]), depth_cfg["valid_range_m"])
            aligned = align_depth_to_rgb(
                z16, float(depth_cfg["depth_scale_m"]), depth_intrinsics, rgb_intrinsics,
                sensor["calibration"]["depth_to_rgb"]["translation_m"],
                sensor["calibration"]["depth_to_rgb"]["rotation_xyzw"],
                rgb_cfg["resolution"],
            )
            publisher.publish_images(endpoint.sim_time, rgb, aligned)
        ready = (
            publisher.color_pub.get_subscription_count() >= 1
            and publisher.depth_pub.get_subscription_count() >= 2
            and endpoint.goal_pub.get_subscription_count() >= 1
            and endpoint.reset_pub.get_subscription_count() >= 3
            and endpoint.safe_stamp > float("-inf")
        )
        if ready:
            break
    if not ready:
        raise RuntimeError("robotics closed-loop nodes did not establish the required ROS graph")

    print("MNS_RESEARCH_NAVIGATION_READY=" + json.dumps({
        "robot_id": ROBOT_ID,
        "scene_id": scene["scene_id"],
        "episodes": episode_ids,
        "research_forest_builder": "build_research_forest",
        "scene_load_time_s": scene_load_time,
        "rgb_resolution": rgb_cfg["resolution"],
        "depth_raw_resolution": depth_cfg["resolution"],
        "registered_depth_resolution": sensor["application_profile"]["depth_aligned_to_rgb"]["resolution"],
        "mapping_enabled": False,
    }, sort_keys=True), flush=True)

    reports = []
    robot_radius = float(robot["surrogate"]["footprint"]["collision_check_radius_m"])
    for episode_id in episode_ids:
        if stop_requested:
            break
        plan = plans[episode_id]
        run_directory = ARGS.run_root / ARGS.run_id / scene["scene_id"] / episode_id
        artifact_directory = ARGS.artifact_root / ARGS.artifact_group / episode_id
        run_directory.mkdir(parents=True, exist_ok=True)
        endpoint.clear_episode()
        endpoint.publish_reset()
        for _ in range(max(3, int(round(0.25 / physics_dt)))):
            endpoint.sim_time = global_frame * physics_dt
            rclpy.spin_once(node, timeout_sec=0.0)
            simulation.step(render=False)
            rgb_camera.update(physics_dt)
            depth_camera.update(physics_dt)
            global_frame += 1
            endpoint.sim_time = global_frame * physics_dt
            publisher.publish_clock(endpoint.sim_time)
        endpoint.publish_episode(episode_id)

        x, y, _, yaw = (float(value) for value in plan["start_pose_xyzyaw"])
        mount_variation = episode_mount_variation(robot, int(scene["scene_seed"]), episode_id)
        rig = place_rig(x, y, yaw, endpoint.sim_time, mount_variation)
        rgb_camera.reset()
        depth_camera.reset()
        endpoint.publish_goal(plan["goal_pose_xyz"], publisher.odom_frame)
        episode_start_global = endpoint.sim_time
        episode_wall_start = time.monotonic()
        initial_goal_distance = math.hypot(
            float(plan["goal_pose_xyz"][0]) - x, float(plan["goal_pose_xyz"][1]) - y
        )
        previous_goal_distance = initial_goal_distance
        minimum_goal_distance = initial_goal_distance
        previous_v = previous_w = 0.0
        previous_rgb_hash = None
        moving_stale_frames = 0
        sensor_frames = 0
        state_trace = []
        command_trace = []
        goal_trace = deque()
        minimum_clearance = float("inf")
        first_collision_tree = None
        first_collision_time = None
        collision = False
        failure_reason = ""
        history_ready_time = None
        override_samples = 0
        override_count = 0
        stop_override_samples = 0
        turn_override_samples = 0
        previous_override = False
        review_frames = {}
        episode_steps = int(math.ceil(ARGS.maximum_duration * ARGS.physics_rate))
        for episode_frame in range(episode_steps):
            endpoint.sim_time = global_frame * physics_dt
            rclpy.spin_once(node, timeout_sec=0.0)
            desired_v, desired_w = endpoint.bounded_safe_command(
                endpoint.sim_time, robot, ARGS.command_timeout
            )
            max_dv = float(robot["motion"]["max_linear_acceleration_mps2"]) * physics_dt
            max_dw = float(robot["motion"]["max_yaw_acceleration_radps2"]) * physics_dt
            command_v = previous_v + float(np.clip(desired_v - previous_v, -max_dv, max_dv))
            command_w = previous_w + float(np.clip(desired_w - previous_w, -max_dw, max_dw))
            previous_v, previous_w = command_v, command_w
            yaw = math.atan2(math.sin(yaw + command_w * physics_dt), math.cos(yaw + command_w * physics_dt))
            _, tangent_frame = surface_frame(built.surface, x, y, yaw)
            x += command_v * float(tangent_frame[0, 0]) * physics_dt
            y += command_v * float(tangent_frame[1, 0]) * physics_dt
            episode_elapsed = endpoint.sim_time - episode_start_global
            rig = place_rig(x, y, yaw, endpoint.sim_time, mount_variation)
            render = (global_frame + 1) % sensor_interval == 0
            simulation.step(render=render)
            rgb_camera.update(physics_dt)
            depth_camera.update(physics_dt)
            global_frame += 1
            endpoint.sim_time = global_frame * physics_dt
            publisher.publish_clock(endpoint.sim_time)
            if global_frame % state_interval == 0:
                publisher.publish_pose(
                    endpoint.sim_time,
                    [x, y, rig.ground_z],
                    rig.body_xyzw,
                    [command_v, 0.0, 0.0],
                    [0.0, 0.0, command_w],
                    rig.body_to_optical_translation,
                    rig.body_to_optical_xyzw,
                )
                navigation = endpoint.navigation_command if endpoint.sim_time - endpoint.navigation_stamp <= ARGS.command_timeout else [0.0, 0.0, 0.0]
                safe = endpoint.safe_command if endpoint.sim_time - endpoint.safe_stamp <= ARGS.command_timeout else [0.0, 0.0, 0.0]
                altered = bool(np.linalg.norm(np.asarray(navigation) - np.asarray(safe)) > 1.0e-3)
                if altered and not previous_override:
                    override_count += 1
                override_samples += int(altered)
                stop_override_samples += int(altered and abs(safe[0]) < 0.02 and navigation[0] > 0.05)
                turn_override_samples += int(altered and abs(safe[2] - navigation[2]) > 0.05)
                previous_override = altered
                goal_distance = math.hypot(
                    float(plan["goal_pose_xyz"][0]) - x, float(plan["goal_pose_xyz"][1]) - y
                )
                minimum_goal_distance = min(minimum_goal_distance, goal_distance)
                state_trace.append([
                    episode_elapsed, x, y, rig.ground_z, *rig.body_xyzw,
                    command_v, command_w, goal_distance,
                ])
                command_trace.append([
                    episode_elapsed, *navigation, *safe, int(altered)
                ])
                goal_trace.append((episode_elapsed, x, y, goal_distance, abs(command_v)))
                while goal_trace and goal_trace[0][0] < episode_elapsed - 8.0:
                    goal_trace.popleft()
                previous_goal_distance = goal_distance
            if render:
                rgb = rgb_camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
                raw = depth_camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy().squeeze().astype(np.float32)
                z16, quantization = quantize_depth_z16(
                    raw, float(depth_cfg["depth_scale_m"]), depth_cfg["valid_range_m"]
                )
                if quantization["saturation_count"]:
                    failure_reason = "sensor_depth_saturation"
                aligned = align_depth_to_rgb(
                    z16, float(depth_cfg["depth_scale_m"]), depth_intrinsics, rgb_intrinsics,
                    sensor["calibration"]["depth_to_rgb"]["translation_m"],
                    sensor["calibration"]["depth_to_rgb"]["rotation_xyzw"],
                    rgb_cfg["resolution"],
                )
                publisher.publish_images(endpoint.sim_time, rgb, aligned)
                sensor_frames += 1
                rgb_hash = hashlib.sha256(rgb.tobytes()).digest()
                if previous_rgb_hash == rgb_hash and abs(command_v) > 0.05:
                    moving_stale_frames += 1
                previous_rgb_hash = rgb_hash
                if ARGS.capture_review:
                    if "start" not in review_frames:
                        review_frames["start"] = (rgb.copy(), aligned.copy(), episode_elapsed)
                    if "middle" not in review_frames and previous_goal_distance <= initial_goal_distance * 0.55:
                        review_frames["middle"] = (rgb.copy(), aligned.copy(), episode_elapsed)
                    review_frames["final"] = (rgb.copy(), aligned.copy(), episode_elapsed)

            clearance, tree_id = _minimum_clearance(scene, x, y, robot_radius)
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance <= 0.0:
                collision = True
                first_collision_tree = tree_id
                first_collision_time = episode_elapsed
                failure_reason = "collision"

            new_diagnostics = endpoint.planning_diagnostics
            if history_ready_time is None and new_diagnostics:
                passing = [item for item in new_diagnostics if item.get("status") == "PASS"]
                if passing:
                    history_ready_time = float(passing[0]["received_sim_time_s"] - episode_start_global)
            for diagnostic in new_diagnostics:
                if "control_minimum_clearance_m" not in diagnostic and diagnostic.get("status") == "PASS":
                    diagnostic["full_minimum_clearance_m"] = _trajectory_clearance(
                        scene, diagnostic.get("full_world_points", []), robot_radius
                    )
                    diagnostic["control_minimum_clearance_m"] = _trajectory_clearance(
                        scene, diagnostic.get("control_world_points", []), robot_radius
                    )
            if any(item.get("status") == "FAIL" for item in new_diagnostics):
                failure_reason = "model_invalid_trajectory"
            if episode_elapsed > 6.0 and not new_diagnostics:
                failure_reason = "history_timeout"
            if len(goal_trace) >= 2 and goal_trace[-1][0] - goal_trace[0][0] >= 7.8:
                displacement = math.hypot(goal_trace[-1][1] - goal_trace[0][1], goal_trace[-1][2] - goal_trace[0][2])
                improvement = goal_trace[0][3] - goal_trace[-1][3]
                moving_fraction = sum(item[4] > 0.08 for item in goal_trace) / len(goal_trace)
                if displacement < 0.12 and improvement < 0.12 and moving_fraction > 0.35:
                    failure_reason = "stall"
            goal_distance = math.hypot(
                float(plan["goal_pose_xyz"][0]) - x, float(plan["goal_pose_xyz"][1]) - y
            )
            if goal_distance <= 0.35:
                break
            if failure_reason:
                break
        else:
            failure_reason = "timeout"

        wall_duration = time.monotonic() - episode_wall_start
        final_goal_distance = math.hypot(
            float(plan["goal_pose_xyz"][0]) - x, float(plan["goal_pose_xyz"][1]) - y
        )
        success = not failure_reason and final_goal_distance <= 0.35
        state_array = np.asarray(state_trace, dtype=np.float64)
        executed_length = (
            float(np.linalg.norm(np.diff(state_array[:, 1:3], axis=0), axis=1).sum())
            if len(state_array) > 1 else 0.0
        )
        simulated_duration = float(state_array[-1, 0]) if len(state_array) else 0.0
        override_fraction = float(override_samples / max(1, len(command_trace)))
        inference = [
            float(item["inference_ms"])
            for item in endpoint.planning_diagnostics
            if item.get("status") == "PASS" and math.isfinite(float(item.get("inference_ms", float("nan"))))
        ]
        for diagnostic in endpoint.planning_diagnostics:
            diagnostic["timestamp_from_episode_start_s"] = float(
                diagnostic.get("received_sim_time_s", episode_start_global) - episode_start_global
            )
        failure_class = _classify_failure(
            failure_reason, endpoint.planning_diagnostics, override_fraction
        )
        report = {
            "report_version": 1,
            "episode_id": episode_id,
            "scene_id": scene["scene_id"],
            "scene_hash": scene["content_hash"],
            "plan_hash": plan["plan_hash"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "success": success,
            "failure_reason": failure_reason or None,
            "failure_class": failure_class,
            "system_success_model_risk": bool(success and override_fraction > 0.25),
            "goal_error_m": final_goal_distance,
            "planned_expert_length_m": float(plan["planned_path_length_m"]),
            "executed_length_m": executed_length,
            "path_efficiency": float(plan["planned_path_length_m"]) / executed_length if executed_length > 0 else None,
            "simulated_duration_s": simulated_duration,
            "wall_duration_s": wall_duration,
            "plan_count": len(inference),
            "planning_rate_hz": len(inference) / simulated_duration if simulated_duration > 0 else 0.0,
            "inference_ms_mean": float(np.mean(inference)) if inference else None,
            "inference_ms_p95": _percentile(inference, 95.0),
            "inference_ms_max": max(inference) if inference else None,
            "collision": collision,
            "collision_tree_id": first_collision_tree,
            "first_collision_timestamp_s": first_collision_time,
            "minimum_clearance_m": minimum_clearance,
            "safety_override_count": override_count,
            "safety_override_duration_s": override_samples / state_rate,
            "safety_override_fraction": override_fraction,
            "safety_stop_override_duration_s": stop_override_samples / state_rate,
            "safety_turn_override_duration_s": turn_override_samples / state_rate,
            "history_ready_time_s": history_ready_time,
            "final_goal_distance_m": final_goal_distance,
            "minimum_goal_distance_m": minimum_goal_distance,
            "sensor_frames": sensor_frames,
            "moving_stale_rgb_frames": moving_stale_frames,
            "robot_id": ROBOT_ID,
            "robot_profile": robot["profile_id"],
            "sensor_profile": sensor["profile_id"],
            "rgb_resolution": rgb_cfg["resolution"],
            "depth_raw_resolution": depth_cfg["resolution"],
            "registered_depth_resolution": sensor["application_profile"]["depth_aligned_to_rgb"]["resolution"],
            "depth_registration": sensor["application_profile"]["depth_aligned_to_rgb"]["policy"],
            "mapping_enabled": False,
            "test_split_used": False,
            "expert_path_used_for_control": False,
            "control_waypoints": 8,
            "prediction_waypoints": 32,
            "full_and_control_same_sample": all(
                len(item.get("full_world_points", [])) == 32
                and item.get("control_world_points", []) == item.get("full_world_points", [])[:8]
                for item in endpoint.planning_diagnostics if item.get("status") == "PASS"
            ),
            "camera_mount_episode_variation": mount_variation,
            "scene_load_time_s": scene_load_time,
            "app_startup_time_s": APP_READY - PROCESS_START,
            "runtime_wall_start_unix_s": time.time() - wall_duration,
        }
        raw_trace = {
            "report": report,
            "expert_reference_path": plan["planned_path"],
            "state_columns": ["episode_time_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "v_mps", "w_radps", "goal_distance_m"],
            "state": state_trace,
            "command_columns": ["episode_time_s", "navigation_v", "navigation_y", "navigation_w", "safe_v", "safe_y", "safe_w", "altered"],
            "commands": command_trace,
            "planning": endpoint.planning_diagnostics,
        }
        _write_json(run_directory / "trace.json", raw_trace)
        _write_json(artifact_directory / "report.json", report)
        if ARGS.capture_review and review_frames:
            names = [name for name in ("start", "middle", "final") if name in review_frames]
            np.savez_compressed(
                run_directory / "review_frames.npz",
                names=np.asarray(names),
                timestamps=np.asarray([review_frames[name][2] for name in names]),
                rgb=np.stack([review_frames[name][0] for name in names]),
                depth=np.stack([review_frames[name][1] for name in names]),
            )
        reports.append(report)
        print("MNS_RESEARCH_NAVIGATION_EPISODE=" + json.dumps(report, sort_keys=True), flush=True)
        endpoint.publish_reset()

    aggregate = {
        "report_version": 1,
        "run_id": ARGS.run_id,
        "artifact_group": ARGS.artifact_group,
        "scene_id": scene["scene_id"],
        "scene_hash": scene["content_hash"],
        "episodes": reports,
        "episode_count": len(reports),
        "success_count": sum(item["success"] for item in reports),
        "collision_count": sum(item["collision"] for item in reports),
        "all_pass": len(reports) == len(episode_ids) and all(item["success"] for item in reports),
        "scene_load_time_s": scene_load_time,
        "total_wall_time_s": time.monotonic() - PROCESS_START,
        "mapping_enabled": False,
        "test_split_used": False,
        "exit_reason": exit_reason,
    }
    _write_json(ARGS.run_root / ARGS.run_id / scene["scene_id"] / "session_report.json", aggregate)
    _write_json(ARGS.artifact_root / ARGS.artifact_group / f"{scene['scene_id']}_session.json", aggregate)
    print("MNS_RESEARCH_NAVIGATION_RESULT=" + json.dumps(aggregate, sort_keys=True), flush=True)

    if rclpy.ok():
        node.destroy_node()
    rclpy.try_shutdown()
    if stop_requested:
        return 130
    if ARGS.fail_on_episode_failure and not aggregate["all_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        APP.close(wait_for_replicator=False, skip_cleanup=True)
    sys.exit(exit_code)
