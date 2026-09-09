"""Shared-world Isaac runtime for namespaced Go2 and UAV instances.

Isaac owns rendering and low-level actuation.  Every external boundary remains
standard ROS 2: registered RGB/depth/semantics, CameraInfo, TF, odometry,
clock, and safe Twist commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("phase1_go2", "phase2_team"), default="phase1_go2")
    parser.add_argument("--system-config", type=Path)
    parser.add_argument("--go2-backend", choices=("rl", "kinematic"), default="rl")
    parser.add_argument("--go2-checkpoint", type=Path, default=Path("/workspace/models/go2_locomotion.pt"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height-px", type=int, default=240)
    parser.add_argument("--sensor-rate", type=float, default=10.0)
    parser.add_argument("--physics-rate", type=float, default=200.0)
    parser.add_argument("--policy-rate", type=float, default=50.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


def _yaw_from_wxyz(quaternion) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    # Isaac imports must follow AppLauncher construction.
    import numpy as np
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.replicator.core")
    enable_extension("isaacsim.ros2.bridge")
    import omni.replicator.core as rep
    import omni.usd
    import torch
    import yaml
    from pxr import Gf, Semantics, UsdGeom
    import isaacsim.core.utils.prims as prim_utils
    import rclpy
    from geometry_msgs.msg import Twist

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext
    from isaaclab_assets import CRAZYFLIE_CFG, UNITREE_GO2_CFG

    from mns_simulation.go2_policy import Go2VelocityPolicy
    from mns_simulation.ros_publisher import StandardRobotPublisher, remap_semantic_ids

    if min(ARGS.sensor_rate, ARGS.physics_rate, ARGS.policy_rate) <= 0:
        raise ValueError("physics, policy, and sensor rates must be positive")
    sensor_interval = round(ARGS.physics_rate / ARGS.sensor_rate)
    policy_interval = round(ARGS.physics_rate / ARGS.policy_rate)
    if not math.isclose(sensor_interval * ARGS.sensor_rate, ARGS.physics_rate) or not math.isclose(
        policy_interval * ARGS.policy_rate, ARGS.physics_rate
    ):
        raise ValueError("sensor_rate and policy_rate must divide physics_rate exactly")

    config_path = ARGS.system_config or Path(
        "/workspace/config/robots/phase1.yaml"
        if ARGS.scenario == "phase1_go2"
        else "/workspace/config/robots/phase2.yaml"
    )
    with config_path.open("r", encoding="utf-8") as stream:
        robot_items = yaml.safe_load(stream)["robots"]
    if not robot_items:
        raise ValueError("system configuration has no robots")
    go2_items = [item for item in robot_items if item["kind"] == "go2"]
    uav_items = [item for item in robot_items if item["kind"] == "uav"]
    if ARGS.go2_backend == "rl" and go2_items and not ARGS.go2_checkpoint.is_file():
        raise FileNotFoundError(f"Go2 RL checkpoint is missing: {ARGS.go2_checkpoint}")

    simulation = SimulationContext(sim_utils.SimulationCfg(
        dt=1.0 / ARGS.physics_rate,
        render_interval=sensor_interval,
        device=ARGS.device,
    ))
    simulation.set_camera_view([18.0, 18.0, 14.0], [0.0, 0.0, 0.0])
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
    light_cfg.func("/World/Light", light_cfg)
    stage = omni.usd.get_context().get_stage()

    def add_semantics(path: str, label: str) -> None:
        prim = stage.GetPrimAtPath(path)
        semantic = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        semantic.CreateSemanticTypeAttr().Set("class")
        semantic.CreateSemanticDataAttr().Set(label)

    add_semantics("/World/Ground", "ground")
    for index, (x, y) in enumerate(((-2.5, -1.0), (2.0, 2.5), (4.5, -3.0), (-4.0, 4.0))):
        path = f"/World/Obstacles/Tree_{index}"
        cylinder = UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateRadiusAttr(0.35)
        cylinder.CreateHeightAttr(5.0)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(x, y, 2.5))
        add_semantics(path, "tree_trunk")

    def spawn_articulation_group(name: str, count: int, template, disable_gravity: bool):
        if count == 0:
            return None
        for index in range(count):
            prim_utils.create_prim(f"/World/{name}/instance_{index}", "Xform")
        config = template.replace(prim_path=f"/World/{name}/instance_.*/Model")
        config.spawn.rigid_props.disable_gravity = disable_gravity
        return Articulation(config)

    go2 = spawn_articulation_group(
        "Go2Instances", len(go2_items), UNITREE_GO2_CFG, ARGS.go2_backend == "kinematic"
    )
    # UAVs are deliberately kinematic sensor platforms in Phase 2.
    uav = spawn_articulation_group("UavInstances", len(uav_items), CRAZYFLIE_CFG, True)

    @dataclass
    class Endpoint:
        item: dict
        group: str
        index: int
        node: object
        publisher: StandardRobotPublisher
        command: list[float]
        render_product: object
        rgb: object
        depth: object
        semantic: object
        pose: list[float]

        def receive_command(self, message: Twist) -> None:
            self.command[:] = message.linear.x, message.linear.y, message.angular.z

        def publish_images(self, sim_time: float) -> None:
            rgb = np.asarray(self.rgb.get_data())
            depth = np.asarray(self.depth.get_data())
            semantic_data = self.semantic.get_data()
            if rgb.size == 0 or depth.size == 0 or not isinstance(semantic_data, dict):
                return
            raw = np.asarray(semantic_data.get("data"))
            info = semantic_data.get("info", {})
            id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
            labels = remap_semantic_ids(raw, id_to_labels)
            self.publisher.publish_images(sim_time, rgb[..., :3], depth, labels)

    rclpy.init(args=None)
    endpoints: list[Endpoint] = []

    def make_endpoint(item: dict, group: str, index: int, publish_clock: bool) -> Endpoint:
        model_path = f"/World/{group}/instance_{index}/Model"
        add_semantics(model_path, "robot")
        camera_path = f"{model_path}/Camera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(22.5)
        camera_xform = UsdGeom.Xformable(camera)
        if item["kind"] == "uav":
            camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.08))
            camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 180.0, 0.0))
        else:
            camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.0, 0.25))
            camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, -90.0))
        render_product = rep.create.render_product(camera_path, (ARGS.width, ARGS.height_px))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        semantic = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False}
        )
        for annotator in (rgb, depth, semantic):
            annotator.attach(render_product)
        robot_id = item["id"]
        node = rclpy.create_node(f"{robot_id}_isaac_bridge", namespace=robot_id)
        publisher = StandardRobotPublisher(
            node,
            robot_id,
            width=ARGS.width,
            height=ARGS.height_px,
            focal_length_px=ARGS.width * 18.0 / 22.5,
            publish_clock=publish_clock,
            camera_mount="downward" if item["kind"] == "uav" else "forward",
        )
        endpoint = Endpoint(
            item=item,
            group=group,
            index=index,
            node=node,
            publisher=publisher,
            command=[0.0, 0.0, 0.0],
            render_product=render_product,
            rgb=rgb,
            depth=depth,
            semantic=semantic,
            pose=[float(value) for value in item["spawn"]],
        )
        node.create_subscription(Twist, "cmd_vel_safe", endpoint.receive_command, 10)
        return endpoint

    clock_assigned = False
    go2_endpoints = []
    for index, item in enumerate(go2_items):
        endpoint = make_endpoint(item, "Go2Instances", index, not clock_assigned)
        clock_assigned = True
        endpoints.append(endpoint)
        go2_endpoints.append(endpoint)
    uav_endpoints = []
    for index, item in enumerate(uav_items):
        endpoint = make_endpoint(item, "UavInstances", index, not clock_assigned)
        clock_assigned = True
        endpoints.append(endpoint)
        uav_endpoints.append(endpoint)

    simulation.reset()

    def initialize_group(entity, group_endpoints: list[Endpoint]) -> None:
        if entity is None:
            return
        if entity.num_instances != len(group_endpoints):
            raise RuntimeError(
                f"Isaac spawned {entity.num_instances} instances for {len(group_endpoints)} configured robots"
            )
        root_state = entity.data.default_root_state.clone()
        for index, endpoint in enumerate(group_endpoints):
            x, y, z, yaw = endpoint.pose
            root_state[index, :3] = torch.tensor((x, y, z), device=simulation.device)
            root_state[index, 3:7] = torch.tensor(
                (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
                device=simulation.device,
            )
            root_state[index, 7:] = 0.0
        entity.write_root_pose_to_sim(root_state[:, :7])
        entity.write_root_velocity_to_sim(root_state[:, 7:])
        entity.write_joint_state_to_sim(
            entity.data.default_joint_pos.clone(), entity.data.default_joint_vel.clone()
        )
        entity.reset()

    initialize_group(go2, go2_endpoints)
    initialize_group(uav, uav_endpoints)
    policy = (
        Go2VelocityPolicy(str(ARGS.go2_checkpoint), ARGS.device)
        if ARGS.go2_backend == "rl" and go2 is not None
        else None
    )
    if policy is not None:
        policy.reset(go2.data)

    physics_dt = 1.0 / ARGS.physics_rate
    frame = 0
    try:
        while APP.is_running():
            for endpoint in endpoints:
                rclpy.spin_once(endpoint.node, timeout_sec=0.0)

            if go2 is not None:
                if policy is not None and frame % policy_interval == 0:
                    commands = torch.tensor(
                        [endpoint.command for endpoint in go2_endpoints],
                        dtype=go2.data.joint_pos.dtype,
                        device=simulation.device,
                    )
                    go2.set_joint_position_target(policy.joint_targets(go2.data, commands))
                elif policy is None:
                    root_pose = go2.data.root_pose_w.clone()
                    root_velocity = torch.zeros(
                        (len(go2_endpoints), 6), dtype=root_pose.dtype, device=simulation.device
                    )
                    for index, endpoint in enumerate(go2_endpoints):
                        vx, vy, omega = endpoint.command
                        cosine, sine = math.cos(endpoint.pose[3]), math.sin(endpoint.pose[3])
                        world_vx, world_vy = cosine * vx - sine * vy, sine * vx + cosine * vy
                        endpoint.pose[0] += world_vx * physics_dt
                        endpoint.pose[1] += world_vy * physics_dt
                        endpoint.pose[3] += omega * physics_dt
                        root_pose[index, :3] = torch.tensor(endpoint.pose[:3], device=simulation.device)
                        root_pose[index, 3:7] = torch.tensor(
                            (math.cos(endpoint.pose[3] / 2.0), 0.0, 0.0, math.sin(endpoint.pose[3] / 2.0)),
                            device=simulation.device,
                        )
                        root_velocity[index] = torch.tensor(
                            (world_vx, world_vy, 0.0, 0.0, 0.0, omega), device=simulation.device
                        )
                    go2.write_root_pose_to_sim(root_pose)
                    go2.write_root_velocity_to_sim(root_velocity)
                go2.write_data_to_sim()

            if uav is not None:
                root_pose = uav.data.root_pose_w.clone()
                root_velocity = torch.zeros(
                    (len(uav_endpoints), 6), dtype=root_pose.dtype, device=simulation.device
                )
                for index, endpoint in enumerate(uav_endpoints):
                    vx, vy, omega = endpoint.command
                    cosine, sine = math.cos(endpoint.pose[3]), math.sin(endpoint.pose[3])
                    world_vx, world_vy = cosine * vx - sine * vy, sine * vx + cosine * vy
                    endpoint.pose[0] += world_vx * physics_dt
                    endpoint.pose[1] += world_vy * physics_dt
                    endpoint.pose[3] += omega * physics_dt
                    root_pose[index, :3] = torch.tensor(endpoint.pose[:3], device=simulation.device)
                    root_pose[index, 3:7] = torch.tensor(
                        (math.cos(endpoint.pose[3] / 2.0), 0.0, 0.0, math.sin(endpoint.pose[3] / 2.0)),
                        device=simulation.device,
                    )
                    root_velocity[index] = torch.tensor(
                        (world_vx, world_vy, 0.0, 0.0, 0.0, omega), device=simulation.device
                    )
                uav.write_root_pose_to_sim(root_pose)
                uav.write_root_velocity_to_sim(root_velocity)
                uav.write_data_to_sim()

            render = frame % sensor_interval == 0
            simulation.step(render=render)
            if go2 is not None:
                go2.update(physics_dt)
            if uav is not None:
                uav.update(physics_dt)
            sim_time = (frame + 1) * physics_dt

            for index, endpoint in enumerate(go2_endpoints):
                position = tuple(float(value) for value in go2.data.root_pos_w[index])
                yaw = _yaw_from_wxyz(go2.data.root_quat_w[index])
                velocity = (
                    float(go2.data.root_lin_vel_b[index, 0]),
                    float(go2.data.root_lin_vel_b[index, 1]),
                    float(go2.data.root_ang_vel_b[index, 2]),
                )
                endpoint.publisher.publish_pose(sim_time, position, yaw, velocity)
            for endpoint in uav_endpoints:
                endpoint.publisher.publish_pose(
                    sim_time, tuple(endpoint.pose[:3]), endpoint.pose[3], tuple(endpoint.command)
                )
            if render:
                for endpoint in endpoints:
                    endpoint.publish_images(sim_time)
            frame += 1
    finally:
        if rclpy.ok():
            for endpoint in endpoints:
                endpoint.node.destroy_node()
        rclpy.try_shutdown()
        APP.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
