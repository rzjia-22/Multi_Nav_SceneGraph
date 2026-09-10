"""Isaac-backed Dataset V0 preview collection runtime.

This runtime intentionally does not require ROS 2 or Hydra.  It executes the
privileged expert with a lightweight non-holonomic surrogate and records the
model-agnostic raw episode schema from real RTX-rendered Isaac cameras.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-rate", type=float, default=100.0)
    parser.add_argument("--maximum-duration", type=float, default=30.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


def main() -> int:
    import h5py  # noqa: F401 - fail before collection if the image lacks schema support
    import numpy as np
    import torch
    import yaml
    import omni.usd
    from pxr import Gf, Semantics, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationContext

    from research_data.common import ROOT, camera_intrinsics, load_yaml, terrain_height
    from research_data.episode import write_episode

    scene = load_yaml(ARGS.scene)
    plan = load_yaml(ARGS.plan)
    robot = load_yaml(ROOT / "config/robots/diablo_standing.yaml")
    sensor = load_yaml(ROOT / "config/sensors/d435i_navigation_v0.yaml")
    if plan["scene_id"] != scene["scene_id"]:
        raise ValueError("episode plan does not match scene")
    if not ARGS.enable_cameras:
        raise ValueError("Dataset V0 collection requires --enable_cameras")
    physics_rate = float(ARGS.physics_rate)
    physics_dt = 1.0 / physics_rate
    sensor_rate = float(sensor["application_profile"]["logging_rate_hz"])
    state_rate = float(sensor["application_profile"]["state_logging_rate_hz"])
    imu_rate = float(sensor["application_profile"]["imu"]["logging_rate_hz"])
    sensor_interval = round(physics_rate / sensor_rate)
    state_interval = round(physics_rate / state_rate)
    imu_interval = round(physics_rate / imu_rate)
    if any(not math.isclose(round(physics_rate / rate) * rate, physics_rate) for rate in (sensor_rate, state_rate, imu_rate)):
        raise ValueError("sensor, state and IMU rates must divide the physics rate")

    material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
        static_friction=float(scene["physics"]["static_friction"]),
        dynamic_friction=float(scene["physics"]["dynamic_friction"]),
        restitution=float(scene["physics"]["restitution"]),
    )
    simulation = SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, render_interval=1, device=ARGS.device, physics_material=material))
    simulation.set_camera_view([13.0, 13.0, 12.0], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    def add_semantics(path: str, label: str) -> None:
        semantic = Semantics.SemanticsAPI.Apply(stage.GetPrimAtPath(path), "Semantics")
        semantic.CreateSemanticTypeAttr().Set("class")
        semantic.CreateSemanticDataAttr().Set(label)

    # Deterministic terrain mesh from the versioned wave parameters.
    resolution = float(scene["terrain"]["mesh_resolution_m"])
    half_x, half_y = float(scene["extent_m"][0]) / 2.0, float(scene["extent_m"][1]) / 2.0
    xs = np.arange(-half_x, half_x + resolution / 2.0, resolution)
    ys = np.arange(-half_y, half_y + resolution / 2.0, resolution)
    points = [Gf.Vec3f(float(x), float(y), terrain_height(scene, float(x), float(y))) for y in ys for x in xs]
    indices, counts = [], []
    width = len(xs)
    for row in range(len(ys) - 1):
        for column in range(width - 1):
            a = row * width + column
            indices.extend((a, a + 1, a + width + 1, a, a + width + 1, a + width))
            counts.extend((3, 3))
    terrain = UsdGeom.Mesh.Define(stage, "/World/ResearchForest/Terrain")
    terrain.CreatePointsAttr(points)
    terrain.CreateFaceVertexIndicesAttr(indices)
    terrain.CreateFaceVertexCountsAttr(counts)
    terrain.CreateSubdivisionSchemeAttr("none")
    terrain.CreateDisplayColorAttr([Gf.Vec3f(*scene["ground"]["rgb"])])
    UsdPhysics.CollisionAPI.Apply(terrain.GetPrim())
    add_semantics("/World/ResearchForest/Terrain", "ground")
    ground_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=tuple(float(value) for value in scene["ground"]["rgb"]),
        roughness=float(scene["ground"]["roughness"]), metallic=0.0,
    )
    ground_material.func("/World/Looks/Ground", ground_material)
    sim_utils.bind_visual_material("/World/ResearchForest/Terrain", "/World/Looks/Ground")

    trunk_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.16, 0.045), roughness=0.82, metallic=0.0)
    trunk_material.func("/World/Looks/Trunk", trunk_material)
    foliage_materials = {}
    for tree_type, color in {
        tree["type"]: tuple(float(value) for value in tree["foliage_color"]) for tree in scene["trees"]
    }.items():
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.78, metallic=0.0)
        material_path = f"/World/Looks/Foliage_{tree_type}"
        material.func(material_path, material)
        foliage_materials[tree_type] = material_path

    UsdGeom.Xform.Define(stage, "/World/ResearchForest/Trees")
    for tree in scene["trees"]:
        x, y = tree["position_m"]
        ground_z = float(tree["ground_height_m"])
        height, radius = float(tree["trunk_height_m"]), float(tree["trunk_radius_m"])
        root = f"/World/ResearchForest/Trees/{tree['tree_id']}"
        UsdGeom.Xform.Define(stage, root)
        trunk_path = root + "/Trunk"
        trunk = UsdGeom.Cylinder.Define(stage, trunk_path)
        trunk.CreateAxisAttr("Z")
        trunk.CreateRadiusAttr(radius)
        trunk.CreateHeightAttr(height)
        trunk.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.12, 0.04)])
        trunk.AddTranslateOp().Set(Gf.Vec3d(x, y, ground_z + height / 2.0))
        UsdPhysics.CollisionAPI.Apply(trunk.GetPrim())
        sim_utils.bind_visual_material(trunk_path, "/World/Looks/Trunk")
        add_semantics(trunk_path, "tree_trunk")
        foliage_path = root + "/Foliage"
        foliage = UsdGeom.Sphere.Define(stage, foliage_path)
        foliage.CreateRadiusAttr(float(tree["foliage_radius_m"]))
        tree_type = scene["factors"]["tree_density"]
        color = Gf.Vec3f(0.07, 0.28, 0.04) if tree_type == "high" else Gf.Vec3f(0.10, 0.36, 0.06)
        foliage.CreateDisplayColorAttr([color])
        foliage.AddTranslateOp().Set(Gf.Vec3d(x, y, ground_z + height - 0.25))
        sim_utils.bind_visual_material(foliage_path, foliage_materials[tree["type"]])
        add_semantics(foliage_path, "foliage")

    dome_cfg = sim_utils.DomeLightCfg(intensity=float(scene["lighting"]["ambient_intensity"]), color=(0.82, 0.88, 1.0))
    dome_cfg.func("/World/Ambient", dome_cfg)
    sun_cfg = sim_utils.DistantLightCfg(intensity=float(scene["lighting"]["intensity"]), angle=0.55, color=(1.0, 0.91, 0.78))
    sun_pitch = math.radians(90.0 - float(scene["lighting"]["sun_elevation_deg"]))
    sun_yaw = math.radians(float(scene["lighting"]["sun_azimuth_deg"]))
    sun_orientation = (
        math.cos(sun_pitch / 2.0) * math.cos(sun_yaw / 2.0),
        -math.sin(sun_pitch / 2.0) * math.sin(sun_yaw / 2.0),
        math.sin(sun_pitch / 2.0) * math.cos(sun_yaw / 2.0),
        math.cos(sun_pitch / 2.0) * math.sin(sun_yaw / 2.0),
    )
    sun_cfg.func("/World/Sun", sun_cfg, translation=(0.0, 0.0, 10.0), orientation=sun_orientation)

    # A deliberately simple body prevents locomotion dynamics from contaminating visual-navigation data.
    UsdGeom.Xform.Define(stage, "/World/DiabloSurrogate")
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
    add_semantics("/World/DiabloSurrogate/Body", "robot")

    rgb_cfg = sensor["application_profile"]["rgb"]
    depth_cfg = sensor["application_profile"]["depth_raw"]

    def make_camera(path: str, cfg: dict, data_types: list[str]):
        image_width, image_height = (int(value) for value in cfg["resolution"])
        fov = cfg["fov_deg"]
        focal = 18.0
        horizontal_aperture = 2.0 * focal * math.tan(math.radians(float(fov["horizontal"])) / 2.0)
        vertical_aperture = 2.0 * focal * math.tan(math.radians(float(fov["vertical"])) / 2.0)
        clipping = cfg.get("clipping_range_m", cfg.get("valid_range_m"))
        return Camera(CameraCfg(
            prim_path=path, update_period=1.0 / sensor_rate, height=image_height, width=image_width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal, horizontal_aperture=horizontal_aperture,
                vertical_aperture=vertical_aperture, clipping_range=(float(clipping[0]), float(clipping[1])),
            ),
        ))

    rgb_camera = make_camera("/World/RgbCamera", rgb_cfg, ["rgb"])
    depth_camera = make_camera("/World/DepthCamera", depth_cfg, ["distance_to_image_plane"])
    rgb_usd_camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/RgbCamera"))
    rgb_usd_camera.GetExposureAttr().Set(float(rgb_cfg["simulated_exposure_compensation_stops"]))
    simulation.reset()

    configured_rgb_intrinsics = np.asarray(camera_intrinsics(*rgb_cfg["resolution"], rgb_cfg["fov_deg"]["horizontal"], rgb_cfg["fov_deg"]["vertical"]), dtype=np.float64)
    configured_depth_intrinsics = np.asarray(camera_intrinsics(*depth_cfg["resolution"], depth_cfg["fov_deg"]["horizontal"], depth_cfg["fov_deg"]["vertical"]), dtype=np.float64)
    # Isaac 5.1's pinhole renderer uses square pixels even when a separate
    # vertical aperture is authored.  The chosen application aspect ratios
    # therefore approximate the device's nominal horizontal/vertical FOV pair.
    configured_rgb_intrinsics[1, 1] = configured_rgb_intrinsics[0, 0]
    configured_depth_intrinsics[1, 1] = configured_depth_intrinsics[0, 0]
    rgb_intrinsics = rgb_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    depth_intrinsics = depth_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(rgb_intrinsics, configured_rgb_intrinsics, rtol=2.0e-3, atol=0.6):
        raise RuntimeError(f"Isaac RGB intrinsics differ from the D435i V0 profile: {rgb_intrinsics.tolist()} vs {configured_rgb_intrinsics.tolist()}")
    if not np.allclose(depth_intrinsics, configured_depth_intrinsics, rtol=2.0e-3, atol=0.6):
        raise RuntimeError(f"Isaac depth intrinsics differ from the D435i V0 profile: {depth_intrinsics.tolist()} vs {configured_depth_intrinsics.tolist()}")
    depth_to_rgb_t = np.asarray(sensor["calibration"]["depth_to_rgb"]["translation_m"], dtype=np.float64)
    depth_to_rgb_q = np.asarray(sensor["calibration"]["depth_to_rgb"]["rotation_xyzw"], dtype=np.float64)

    def align_depth(raw: np.ndarray) -> np.ndarray:
        height, width = raw.shape
        vv, uu = np.indices((height, width), dtype=np.float32)
        z = raw.astype(np.float32)
        valid = np.isfinite(z) & (z >= float(depth_cfg["valid_range_m"][0])) & (z <= float(depth_cfg["valid_range_m"][1]))
        x = (uu[valid] - depth_intrinsics[0, 2]) * z[valid] / depth_intrinsics[0, 0] + depth_to_rgb_t[0]
        y = (vv[valid] - depth_intrinsics[1, 2]) * z[valid] / depth_intrinsics[1, 1] + depth_to_rgb_t[1]
        rz = z[valid] + depth_to_rgb_t[2]
        projected_u = np.rint(rgb_intrinsics[0, 0] * x / rz + rgb_intrinsics[0, 2]).astype(np.int64)
        projected_v = np.rint(rgb_intrinsics[1, 1] * y / rz + rgb_intrinsics[1, 2]).astype(np.int64)
        output_height, output_width = rgb_cfg["resolution"][1], rgb_cfg["resolution"][0]
        inside = (projected_u >= 0) & (projected_u < output_width) & (projected_v >= 0) & (projected_v < output_height)
        flat = np.full(output_height * output_width, np.inf, dtype=np.float32)
        np.minimum.at(flat, projected_v[inside] * output_width + projected_u[inside], rz[inside])
        aligned = flat.reshape(output_height, output_width)
        aligned[~np.isfinite(aligned)] = 0.0
        return aligned

    def rotation_matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                w = (matrix[2, 1] - matrix[1, 2]) / scale
                x, y, z = 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale
            elif axis == 1:
                scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                w = (matrix[0, 2] - matrix[2, 0]) / scale
                x, y, z = (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale
            else:
                scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                w = (matrix[1, 0] - matrix[0, 1]) / scale
                x, y, z = (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale
        quaternion = np.asarray([x, y, z, w], dtype=np.float64)
        return quaternion / np.linalg.norm(quaternion)

    path = np.asarray(plan["planned_path"], dtype=np.float64)
    # Densify the smoothed polyline for stable pure-pursuit target selection.
    dense = [path[0]]
    for index in range(1, len(path)):
        distance = np.linalg.norm(path[index, :2] - path[index - 1, :2])
        for fraction in np.linspace(0.0, 1.0, max(2, int(math.ceil(distance / 0.10)) + 1))[1:]:
            point = path[index - 1] + fraction * (path[index] - path[index - 1])
            dense.append(point)
    dense = np.asarray(dense)
    x, y, _, yaw = (float(value) for value in plan["start_pose_xyzyaw"])
    command_v = command_w = 0.0
    previous_v = 0.0
    body_prim = stage.GetPrimAtPath("/World/DiabloSurrogate")
    body_xform = UsdGeom.Xformable(body_prim)
    body_translate = body_xform.AddTranslateOp()
    body_orient = body_xform.AddOrientOp()
    mount = robot["camera_mount"]
    pitch_nominal = math.radians(float(mount["nominal_pitch_deg"]))
    correlation = mount["correlated_motion"]
    payload = {name: [] for name in (
        "timestamp_s", "position_xyz", "orientation_xyzw", "linear_velocity_xyz", "angular_velocity_xyz", "command_vw",
        "imu_timestamp_s", "imu_linear_acceleration_xyz", "imu_angular_velocity_xyz", "sensor_timestamp_s",
        "rgb", "depth_raw_m", "depth_aligned_m", "camera_pose",
    )}
    payload["expert_path"] = path.tolist()
    payload["calibration"] = {
        "rgb_intrinsics": rgb_intrinsics, "depth_intrinsics": depth_intrinsics,
        "depth_to_rgb_translation_m": depth_to_rgb_t, "depth_to_rgb_rotation_xyzw": depth_to_rgb_q,
    }
    success = False
    collision = False
    max_steps = int(ARGS.maximum_duration * physics_rate)
    max_speed = float(robot["motion"]["max_forward_speed_mps"])
    max_yaw_rate = float(robot["motion"]["max_yaw_rate_radps"])
    max_accel = float(robot["motion"]["max_linear_acceleration_mps2"])
    lookahead = float(plan["controller"]["lookahead_m"])
    goal_tolerance = float(plan["controller"]["goal_tolerance_m"])
    body_height = float(robot["surrogate"]["body_height_m"])

    print("MNS_DATASET_RUNTIME_READY=" + json.dumps({"scene_id": scene["scene_id"], "episode_id": plan["episode_id"], "device": str(simulation.device), "rgb_resolution": rgb_cfg["resolution"], "depth_resolution": depth_cfg["resolution"]}, sort_keys=True), flush=True)
    for frame in range(max_steps):
        sim_time = frame * physics_dt
        position = np.asarray([x, y])
        distances = np.linalg.norm(dense[:, :2] - position, axis=1)
        nearest = int(np.argmin(distances))
        target_index = nearest
        while target_index < len(dense) - 1 and np.linalg.norm(dense[target_index, :2] - position) < lookahead:
            target_index += 1
        target = dense[target_index, :2]
        goal_distance = float(np.linalg.norm(path[-1, :2] - position))
        if goal_distance <= goal_tolerance:
            command_v = command_w = 0.0
            success = True
        else:
            desired = math.atan2(target[1] - y, target[0] - x)
            error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
            desired_speed = max_speed * max(0.18, math.cos(error))
            if goal_distance < 0.8:
                desired_speed *= max(0.25, goal_distance / 0.8)
            delta_v = float(np.clip(desired_speed - previous_v, -max_accel * physics_dt, max_accel * physics_dt))
            command_v = previous_v + delta_v
            command_w = float(np.clip(2.1 * error, -max_yaw_rate, max_yaw_rate))
        acceleration = (command_v - previous_v) / physics_dt
        previous_v = command_v
        if not success:
            yaw = math.atan2(math.sin(yaw + command_w * physics_dt), math.cos(yaw + command_w * physics_dt))
            x += command_v * math.cos(yaw) * physics_dt
            y += command_v * math.sin(yaw) * physics_dt
        ground_z = terrain_height(scene, x, y)
        body_translate.Set(Gf.Vec3d(x, y, ground_z))
        body_orient.Set(Gf.Quatf(math.cos(yaw / 2.0), Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2.0))))

        oscillation = math.sin(2.0 * math.pi * float(correlation["frequency_hz"]) * sim_time) if correlation["enabled"] else 0.0
        camera_z = ground_z + float(mount["nominal_height_m"]) + float(correlation["vertical_amplitude_m"]) * oscillation
        pitch = pitch_nominal + math.radians(float(correlation["pitch_amplitude_deg"])) * oscillation
        roll = math.radians(float(correlation["roll_amplitude_deg"])) * oscillation
        forward = np.asarray([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        lateral_left = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0])
        optical_right_zero = -lateral_left
        optical_down_zero = np.cross(forward, optical_right_zero)
        optical_down_zero /= np.linalg.norm(optical_down_zero)
        optical_right = math.cos(roll) * optical_right_zero + math.sin(roll) * optical_down_zero
        optical_down = -math.sin(roll) * optical_right_zero + math.cos(roll) * optical_down_zero
        optical_xyzw = rotation_matrix_to_xyzw(np.column_stack((optical_right, optical_down, forward)))
        optical_wxyz = np.asarray([optical_xyzw[3], optical_xyzw[0], optical_xyzw[1], optical_xyzw[2]])
        rgb_eye = np.asarray([x, y, camera_z]) + float(mount["forward_offset_m"]) * np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
        depth_eye = rgb_eye - depth_to_rgb_t[0] * optical_right
        camera_orientation = torch.as_tensor(np.stack([optical_wxyz]), dtype=torch.float32, device=simulation.device)
        rgb_camera.set_world_poses(torch.as_tensor(np.stack([rgb_eye]), dtype=torch.float32, device=simulation.device), camera_orientation, convention="ros")
        depth_camera.set_world_poses(torch.as_tensor(np.stack([depth_eye]), dtype=torch.float32, device=simulation.device), camera_orientation, convention="ros")

        render_frame = (frame + 1) % sensor_interval == 0
        simulation.step(render=render_frame)
        rgb_camera.update(physics_dt)
        depth_camera.update(physics_dt)
        timestamp = (frame + 1) * physics_dt
        if (frame + 1) % imu_interval == 0:
            payload["imu_timestamp_s"].append(timestamp)
            payload["imu_linear_acceleration_xyz"].append([acceleration, command_v * command_w, 9.80665])
            payload["imu_angular_velocity_xyz"].append([0.0, 0.0, command_w])
        if (frame + 1) % state_interval == 0:
            payload["timestamp_s"].append(timestamp)
            payload["position_xyz"].append([x, y, ground_z])
            payload["orientation_xyzw"].append([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
            payload["linear_velocity_xyz"].append([command_v * math.cos(yaw), command_v * math.sin(yaw), 0.0])
            payload["angular_velocity_xyz"].append([0.0, 0.0, command_w])
            payload["command_vw"].append([command_v, command_w])
        if render_frame:
            rgb = rgb_camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
            raw = depth_camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy().squeeze()
            raw = raw.astype(np.float32)
            raw[~np.isfinite(raw)] = 0.0
            raw[(raw < float(depth_cfg["valid_range_m"][0])) | (raw > float(depth_cfg["valid_range_m"][1]))] = 0.0
            payload["sensor_timestamp_s"].append(timestamp)
            payload["rgb"].append(rgb)
            payload["depth_raw_m"].append(raw)
            payload["depth_aligned_m"].append(align_depth(raw))
            payload["camera_pose"].append([*rgb_eye, *optical_xyzw])
        if success and len(payload["sensor_timestamp_s"]) >= 3:
            break

    if not success:
        raise RuntimeError(f"expert failed to reach goal within {ARGS.maximum_duration:.1f} s")
    positions = np.asarray(payload["position_xyz"])
    executed_length = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
    metadata = {
        "dataset_version": "dataset_v0", "episode_id": plan["episode_id"], "episode_type": plan["episode_type"],
        "split": plan["split"], "scene_id": scene["scene_id"], "scene_seed": scene["scene_seed"],
        "robot_profile": plan["robot_profile"], "sensor_profile": plan["sensor_profile"],
        "start_pose_xyzyaw": plan["start_pose_xyzyaw"], "goal_pose_xyz": plan["goal_pose_xyz"],
        "planned_path_length_m": plan["planned_path_length_m"], "executed_path_length_m": executed_length,
        "success": True, "collision": collision, "planner_type": plan["planner"]["type"],
        "controller_type": plan["controller"]["type"], "configuration_hashes": plan["configuration_hashes"],
        "isaac_sim_version": "5.1.0", "isaac_lab_version": "2.3.1",
        "rgb_depth_alignment": sensor["application_profile"]["depth_aligned_to_rgb"]["policy"],
        "semantic_recorded": False,
    }
    write_episode(ARGS.output, payload, metadata)
    print("MNS_DATASET_RUNTIME_RESULT=" + json.dumps({"status": "PASS", "success": True, "duration_s": payload["timestamp_s"][-1], "rgb_frames": len(payload["rgb"]), "depth_frames": len(payload["depth_raw_m"]), "imu_samples": len(payload["imu_timestamp_s"]), "pose_samples": len(payload["timestamp_s"]), "executed_path_length_m": executed_length, "output": str(ARGS.output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        APP.close(wait_for_replicator=False, skip_cleanup=True)
    sys.exit(code)
