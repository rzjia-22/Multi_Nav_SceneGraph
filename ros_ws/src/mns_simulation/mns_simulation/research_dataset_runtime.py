"""Scene-batched Isaac Research Forest Dataset V0 collection runtime."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

PROCESS_START = time.monotonic()

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--physics-rate", type=float, default=100.0)
    parser.add_argument("--maximum-duration", type=float, default=30.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app
APP_READY = time.monotonic()


class ResourceMonitor:
    """Small local resource sampler with no external monitoring service."""

    def __init__(self):
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        try:
            import psutil

            self.psutil = psutil
            self.process = psutil.Process()
            self.process.cpu_percent(None)
        except ImportError:
            self.psutil = None
            self.process = None

    def start(self):
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=3.0)
        return self.summarize()

    def summarize(self, start: int = 0, end: int | None = None) -> dict:
        samples = self.samples[start:end]
        if not samples:
            return {"samples": 0}
        numeric = sorted({key for sample in samples for key, value in sample.items() if isinstance(value, (int, float))})
        report = {"samples": len(samples)}
        for key in numeric:
            values = [float(sample[key]) for sample in samples if key in sample]
            report[f"{key}_average"] = sum(values) / len(values)
            report[f"{key}_peak"] = max(values)
        for sample in samples:
            if "gpu_name" in sample:
                report["gpu_name"] = sample["gpu_name"]
                report["gpu_memory_total_mib"] = sample["gpu_memory_total_mib"]
                break
        return report

    def snapshot(self) -> dict:
        sample: dict = {"monotonic_time_s": time.monotonic()}
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                text=True, timeout=3,
            ).strip().split(", ")
            sample.update({
                "gpu_name": output[0], "gpu_memory_used_mib": float(output[1]),
                "gpu_memory_total_mib": float(output[2]), "gpu_utilization_percent": float(output[3]),
                "gpu_temperature_c": float(output[4]), "gpu_power_w": float(output[5]),
            })
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        if self.process is not None:
            try:
                memory = self.psutil.virtual_memory()
                sample.update({
                    "process_cpu_percent": float(self.process.cpu_percent(None)),
                    "process_rss_mib": float(self.process.memory_info().rss / 1024**2),
                    "host_memory_used_mib": float(memory.used / 1024**2),
                })
            except self.psutil.Error:
                pass
        self.samples.append(sample)
        return sample

    def _sample_loop(self):
        while not self._stop.wait(0.5):
            self.snapshot()


def rotation_matrix_to_xyzw(matrix):
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w, x = 0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale
        y, z = (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale
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


def surface_frame(surface, x: float, y: float, yaw: float):
    """Return ground height and body x-forward/y-left/z-normal frame."""
    import numpy as np

    height, normal = surface.height_and_normal(x, y)
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    heading = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    forward = heading - normal * float(np.dot(heading, normal))
    forward /= np.linalg.norm(forward)
    left = np.cross(normal, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, normal)
    return height, np.column_stack((forward, left, normal))


def make_camera(sim_utils, Camera, CameraCfg, path: str, cfg: dict, sensor_rate: float, data_types: list[str]):
    width, height = (int(value) for value in cfg["resolution"])
    focal = 18.0
    horizontal_aperture = 2.0 * focal * math.tan(math.radians(float(cfg["fov_deg"]["horizontal"])) / 2.0)
    vertical_aperture = 2.0 * focal * math.tan(math.radians(float(cfg["fov_deg"]["vertical"])) / 2.0)
    clipping = cfg.get("clipping_range_m", cfg.get("valid_range_m"))
    return Camera(CameraCfg(
        prim_path=path, update_period=1.0 / sensor_rate, height=height, width=width, data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal, horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture, clipping_range=(float(clipping[0]), float(clipping[1])),
        ),
    ))


def run_episode(simulation, stage, built, scene, robot, sensor, rgb_camera, depth_camera,
                rgb_intrinsics, depth_intrinsics, episode_id: str, output_directory: Path) -> dict:
    import h5py
    import numpy as np
    import torch
    from pxr import Gf, UsdGeom

    from research_data.common import file_hash, load_yaml, stable_hash
    from research_data.depth import ALIGNMENT_ALGORITHM, ALIGNMENT_VERSION, align_depth_to_rgb, alignment_round_trip_report, dequantize_depth_z16, quantize_depth_z16
    from research_data.episode import COLLECTOR_VERSION, EPISODE_SCHEMA_VERSION, write_episode
    from research_data.expert import PLANNER_VERSION, validate_plan
    from research_data.validation import validate_episode, write_report

    episode_directory = output_directory / episode_id
    episode_directory.mkdir(parents=True, exist_ok=True)
    plan_path = episode_directory / "episode_plan.yaml"
    if not plan_path.is_file():
        raise FileNotFoundError(f"preflighted exact plan is missing: {plan_path}")
    plan = load_yaml(plan_path)
    plan_validation = validate_plan(scene, plan)
    episode_path = episode_directory / "episode.h5"
    temporary_episode_path = episode_directory / ".episode.h5.inprogress"
    if episode_path.exists():
        try:
            prior_validation = validate_episode(episode_path, plan_path)
        except Exception:
            episode_path.unlink()
        else:
            return {
                "status": "SKIPPED_VALID", "episode_id": episode_id, "success": True,
                "planned_path_length_m": prior_validation["planned_path_length_m"],
                "executed_path_length_m": prior_validation["executed_path_length_m"],
                "simulated_duration_s": prior_validation["duration_s"], "wall_execution_time_s": 0.0,
                "real_time_factor": None, "rgb_frames": prior_validation["rgb_frames"],
                "depth_frames": prior_validation["depth_frames"], "imu_samples": prior_validation["imu_samples"],
                "state_samples": prior_validation["pose_samples"], "plan_hash": plan["plan_hash"],
                "planner_version": PLANNER_VERSION, "hdf5_sha256": prior_validation["hdf5_sha256"],
                "hdf5_size_bytes": episode_path.stat().st_size, "validation": prior_validation,
                "episode_path": str(episode_path), "plan_path": str(plan_path),
            }
    if temporary_episode_path.exists():
        temporary_episode_path.unlink()

    physics_rate = float(ARGS.physics_rate)
    physics_dt = 1.0 / physics_rate
    sensor_rate = float(sensor["application_profile"]["logging_rate_hz"])
    state_rate = float(sensor["application_profile"]["state_logging_rate_hz"])
    imu_rate = float(sensor["application_profile"]["imu"]["logging_rate_hz"])
    sensor_interval, state_interval, imu_interval = (round(physics_rate / rate) for rate in (sensor_rate, state_rate, imu_rate))
    if any(not math.isclose(round(physics_rate / rate) * rate, physics_rate) for rate in (sensor_rate, state_rate, imu_rate)):
        raise ValueError("sensor, state and IMU rates must divide the physics rate")

    rgb_cfg = sensor["application_profile"]["rgb"]
    depth_cfg = sensor["application_profile"]["depth_raw"]
    rgb_resolution = tuple(int(value) for value in rgb_cfg["resolution"])
    depth_to_rgb_t = np.asarray(sensor["calibration"]["depth_to_rgb"]["translation_m"], dtype=np.float64)
    depth_to_rgb_q = np.asarray(sensor["calibration"]["depth_to_rgb"]["rotation_xyzw"], dtype=np.float64)
    depth_scale_m = float(depth_cfg["depth_scale_m"])

    path = np.asarray(plan["planned_path"], dtype=np.float64)
    dense = [path[0]]
    for index in range(1, len(path)):
        distance = np.linalg.norm(path[index, :2] - path[index - 1, :2])
        dense.extend(path[index - 1] + fraction * (path[index] - path[index - 1]) for fraction in np.linspace(0.0, 1.0, max(2, int(math.ceil(distance / 0.10)) + 1))[1:])
    dense = np.asarray(dense)
    x, y, _, yaw = (float(value) for value in plan["start_pose_xyzyaw"])
    command_v = command_w = previous_v = 0.0
    body_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/DiabloSurrogate"))
    body_translate, body_orient = body_xform.GetOrderedXformOps()
    mount = robot["camera_mount"]
    rng = np.random.Generator(np.random.PCG64(int(scene["scene_seed"]) + int(episode_id.rsplit("_", 1)[-1]) + 700_001))
    variation = mount["episode_variation"]
    episode_mount = {
        "height_offset_m": float(rng.uniform(*variation["height_m"])),
        "pitch_offset_deg": float(rng.uniform(*variation["pitch_deg"])),
        "roll_offset_deg": float(rng.uniform(*variation["roll_deg"])),
    }
    correlation = mount["correlated_motion"]
    payload = {name: [] for name in (
        "timestamp_s", "position_xyz", "orientation_xyzw", "linear_velocity_xyz", "angular_velocity_xyz", "command_vw",
        "imu_timestamp_s", "imu_linear_acceleration_xyz", "imu_angular_velocity_xyz", "sensor_timestamp_s",
        "rgb", "depth_raw_z16", "camera_pose",
    )}
    payload["expert_path"] = path.tolist()
    payload["calibration"] = {
        "rgb_intrinsics": rgb_intrinsics, "depth_intrinsics": depth_intrinsics,
        "depth_to_rgb_translation_m": depth_to_rgb_t, "depth_to_rgb_rotation_xyzw": depth_to_rgb_q,
        "depth_scale_m": depth_scale_m, "rgb_resolution_wh": rgb_resolution,
        "depth_resolution_wh": tuple(int(value) for value in depth_cfg["resolution"]),
    }
    online_aligned_reference, quantization_errors, quantization_reports = [], [], []
    success = False
    max_steps = int(ARGS.maximum_duration * physics_rate)
    max_speed = float(robot["motion"]["max_forward_speed_mps"])
    max_yaw_rate = float(robot["motion"]["max_yaw_rate_radps"])
    max_accel = float(robot["motion"]["max_linear_acceleration_mps2"])
    lookahead = float(plan["controller"]["lookahead_m"])
    goal_tolerance = float(plan["controller"]["goal_tolerance_m"])

    rgb_camera.reset()
    depth_camera.reset()

    print("MNS_DATASET_EPISODE_READY=" + json.dumps({"scene_id": scene["scene_id"], "episode_id": episode_id}), flush=True)
    execution_started = time.monotonic()
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
            command_v = previous_v + float(np.clip(desired_speed - previous_v, -max_accel * physics_dt, max_accel * physics_dt))
            command_w = float(np.clip(2.1 * error, -max_yaw_rate, max_yaw_rate))
        acceleration = (command_v - previous_v) / physics_dt
        previous_v = command_v
        if not success:
            yaw = math.atan2(math.sin(yaw + command_w * physics_dt), math.cos(yaw + command_w * physics_dt))
            _, current_frame = surface_frame(built.surface, x, y, yaw)
            x += command_v * float(current_frame[0, 0]) * physics_dt
            y += command_v * float(current_frame[1, 0]) * physics_dt

        ground_z, body_rotation = surface_frame(built.surface, x, y, yaw)
        body_xyzw = rotation_matrix_to_xyzw(body_rotation)
        body_translate.Set(Gf.Vec3d(x, y, ground_z))
        body_orient.Set(Gf.Quatf(float(body_xyzw[3]), Gf.Vec3f(*[float(value) for value in body_xyzw[:3]])))
        oscillation = math.sin(2.0 * math.pi * float(correlation["frequency_hz"]) * sim_time) if correlation["enabled"] else 0.0
        camera_height = float(mount["nominal_height_m"]) + episode_mount["height_offset_m"] + float(correlation["vertical_amplitude_m"]) * oscillation
        pitch = math.radians(float(mount["nominal_pitch_deg"]) + episode_mount["pitch_offset_deg"] + float(correlation["pitch_amplitude_deg"]) * oscillation)
        roll = math.radians(float(mount["nominal_roll_deg"]) + episode_mount["roll_offset_deg"] + float(correlation["roll_amplitude_deg"]) * oscillation)
        body_forward, body_left, body_up = body_rotation.T
        optical_forward = math.cos(pitch) * body_forward + math.sin(pitch) * body_up
        optical_right_zero = -body_left
        optical_down_zero = np.cross(optical_forward, optical_right_zero)
        optical_down_zero /= np.linalg.norm(optical_down_zero)
        optical_right = math.cos(roll) * optical_right_zero + math.sin(roll) * optical_down_zero
        optical_down = -math.sin(roll) * optical_right_zero + math.cos(roll) * optical_down_zero
        optical_rotation = np.column_stack((optical_right, optical_down, optical_forward))
        optical_xyzw = rotation_matrix_to_xyzw(optical_rotation)
        optical_wxyz = np.asarray([optical_xyzw[3], *optical_xyzw[:3]])
        rgb_eye = np.asarray([x, y, ground_z]) + body_rotation @ np.asarray([
            float(mount["forward_offset_m"]), float(mount["lateral_offset_m"]), camera_height,
        ])
        depth_eye = rgb_eye - optical_rotation @ depth_to_rgb_t
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
            payload["orientation_xyzw"].append(body_xyzw.tolist())
            payload["linear_velocity_xyz"].append((command_v * body_rotation[:, 0]).tolist())
            payload["angular_velocity_xyz"].append((command_w * body_rotation[:, 2]).tolist())
            payload["command_vw"].append([command_v, command_w])
        if render_frame:
            rgb = rgb_camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
            raw = depth_camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy().squeeze().astype(np.float32)
            encoded, quant_report = quantize_depth_z16(raw, depth_scale_m, depth_cfg["valid_range_m"])
            restored = dequantize_depth_z16(encoded, depth_scale_m)
            valid = encoded != 0
            if valid.any():
                quantization_errors.append(np.abs(restored[valid] - raw[valid]))
            quantization_reports.append(quant_report)
            aligned = align_depth_to_rgb(encoded, depth_scale_m, depth_intrinsics, rgb_intrinsics, depth_to_rgb_t, depth_to_rgb_q, rgb_resolution)
            payload["sensor_timestamp_s"].append(timestamp)
            payload["rgb"].append(rgb)
            payload["depth_raw_z16"].append(encoded)
            payload["camera_pose"].append([*rgb_eye, *optical_xyzw])
            online_aligned_reference.append(aligned)
        if success and len(payload["sensor_timestamp_s"]) >= 3:
            break
    execution_wall_time = time.monotonic() - execution_started
    if not success:
        raise RuntimeError(f"expert failed to reach goal within {ARGS.maximum_duration:.1f} s")

    positions = np.asarray(payload["position_xyz"])
    orientations = np.asarray(payload["orientation_xyzw"])
    executed_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    qx, qy, qz, qw = orientations.T
    roll = np.degrees(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
    pitch_body = np.degrees(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
    surface_height_error = max(abs(float(point[2]) - built.surface.height(float(point[0]), float(point[1]))) for point in positions)
    quant_errors = np.concatenate(quantization_errors) if quantization_errors else np.asarray([], dtype=np.float32)
    quantization = {
        "depth_scale_m": depth_scale_m, "valid_range_m": [float(value) for value in depth_cfg["valid_range_m"]],
        "invalid_convention": "zero_is_invalid", "saturation_count": int(sum(item["saturation_count"] for item in quantization_reports)),
        "mean_absolute_error_m": float(quant_errors.mean()) if quant_errors.size else 0.0,
        "p95_absolute_error_m": float(np.percentile(quant_errors, 95)) if quant_errors.size else 0.0,
        "maximum_absolute_error_m": float(quant_errors.max()) if quant_errors.size else 0.0,
    }
    metadata = {
        "dataset_version": "dataset_v0", "schema_version": EPISODE_SCHEMA_VERSION, "collector_version": COLLECTOR_VERSION,
        "episode_id": episode_id, "episode_type": plan["episode_type"], "split": plan["split"],
        "scene_id": scene["scene_id"], "scene_seed": scene["scene_seed"], "scene_schema_version": scene["schema_version"],
        "scene_content_hash": scene["content_hash"], "robot_profile": plan["robot_profile"], "sensor_profile": plan["sensor_profile"],
        "plan_hash": plan["plan_hash"], "planner_version": plan["planner"]["version"],
        "start_pose_xyzyaw": plan["start_pose_xyzyaw"], "goal_pose_xyz": plan["goal_pose_xyz"],
        "planned_path_length_m": plan["planned_path_length_m"], "executed_path_length_m": executed_length,
        "success": True, "collision": False, "planner_type": plan["planner"]["type"], "controller_type": plan["controller"]["type"],
        "configuration_hashes": {**plan["configuration_hashes"], "asset_registry": stable_hash(load_yaml(ARGS.assets))},
        "isaac_sim_version": "5.1.0", "isaac_lab_version": "2.3.1",
        "rgb_depth_alignment": {"algorithm": ALIGNMENT_ALGORITHM, "version": ALIGNMENT_VERSION, "persistence": "derived_offline"},
        "depth_storage": quantization, "terrain_statistics": built.terrain_statistics,
        "camera_mount_episode_variation": episode_mount, "semantic_recorded": False,
        "training_eligible": True,
    }
    try:
        storage = write_episode(temporary_episode_path, payload, metadata)
        with h5py.File(temporary_episode_path, "r") as episode:
            reconstructed = align_depth_to_rgb(
                np.asarray(episode["sensors/depth_raw_z16"]), float(np.asarray(episode["calibration/depth_scale_m"])),
                np.asarray(episode["calibration/depth_intrinsics"]), np.asarray(episode["calibration/rgb_intrinsics"]),
                np.asarray(episode["calibration/depth_to_rgb_translation_m"]), np.asarray(episode["calibration/depth_to_rgb_rotation_xyzw"]),
                tuple(np.asarray(episode["calibration/rgb_resolution_wh"], dtype=int)),
            )
        alignment = alignment_round_trip_report(np.asarray(online_aligned_reference), reconstructed)
        if alignment["valid_pixel_agreement"] != 1.0 or alignment["maximum_depth_difference_m"] != 0.0:
            raise RuntimeError(f"persisted depth alignment differs from runtime reference: {alignment}")
        validation = validate_episode(temporary_episode_path, plan_path)
        hdf5_sha256 = file_hash(temporary_episode_path)
        os.replace(temporary_episode_path, episode_path)
        validation["hdf5_sha256"] = hdf5_sha256
        write_report(episode_directory / "validation_report.json", validation)
    except BaseException:
        if temporary_episode_path.exists():
            temporary_episode_path.unlink()
        raise
    simulated_duration = float(payload["timestamp_s"][-1] - payload["timestamp_s"][0])
    return {
        "status": "PASS", "episode_id": episode_id, "success": True,
        "planned_path_length_m": float(plan["planned_path_length_m"]), "executed_path_length_m": executed_length,
        "simulated_duration_s": simulated_duration, "wall_execution_time_s": execution_wall_time,
        "real_time_factor": simulated_duration / execution_wall_time, "rgb_frames": len(payload["rgb"]),
        "depth_frames": len(payload["depth_raw_z16"]), "imu_samples": len(payload["imu_timestamp_s"]),
        "state_samples": len(payload["timestamp_s"]), "depth_quantization": quantization,
        "alignment_round_trip": alignment, "storage": storage,
        "plan_validation": plan_validation, "plan_hash": plan["plan_hash"],
        "planner_version": PLANNER_VERSION, "hdf5_sha256": hdf5_sha256,
        "hdf5_size_bytes": episode_path.stat().st_size, "goal_error_m": validation["goal_error_m"],
        "validation": validation,
        "surface_following": {
            "maximum_height_error_m": surface_height_error,
            "body_roll_range_deg": [float(roll.min()), float(roll.max())],
            "body_pitch_range_deg": [float(pitch_body.min()), float(pitch_body.max())],
            "pose_composition": "terrain_normal_then_navigation_yaw_then_camera_mount_then_correlated_motion",
        },
        "episode_path": str(episode_path), "plan_path": str(plan_path),
    }


def _memory_leak_report(endpoints: list[dict]) -> dict:
    report = {"episode_endpoints": endpoints, "suspected": False, "signals": []}
    for key, label in (("process_rss_mib", "RSS"), ("gpu_memory_used_mib", "VRAM")):
        values = [float(item["end"][key]) for item in endpoints if key in item["end"]]
        if len(values) < 3:
            continue
        deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
        monotonic_growth = all(delta > 128.0 for delta in deltas) and values[-1] - values[0] > 768.0
        report[f"{key}_end_values"] = values
        report[f"{key}_net_change"] = values[-1] - values[0]
        report[f"{key}_monotonic_significant_growth"] = monotonic_growth
        if monotonic_growth:
            report["signals"].append(f"{label} grew by more than 128 MiB after every episode and over 768 MiB overall")
    report["suspected"] = bool(report["signals"])
    return report


def main() -> int:
    import numpy as np
    import omni.usd
    from pxr import Gf, Semantics, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationContext
    from research_data.common import ROOT, camera_intrinsics, dump_yaml, load_yaml, scene_paths
    from research_data.episode import COLLECTOR_VERSION, EPISODE_SCHEMA_VERSION
    from research_data.expert import PLANNER_VERSION, sample_episode_plan, validate_plan
    from research_data.validation import write_report
    from mns_simulation.research_forest_scene import build_research_forest, export_stage_snapshot

    scene, registry = load_yaml(ARGS.scene), load_yaml(ARGS.assets)
    robot = load_yaml(ROOT / "config/robots/diablo_standing.yaml")
    sensor = load_yaml(ROOT / "config/sensors/d435i_navigation_v0.yaml")
    ARGS.output_directory.mkdir(parents=True, exist_ok=True)
    simulation = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / float(ARGS.physics_rate), render_interval=1, device=ARGS.device))
    simulation.set_camera_view([13.0, 13.0, 12.0], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    monitor = ResourceMonitor()
    monitor.start()
    session_start_snapshot = monitor.snapshot()
    scene_started = time.monotonic()
    built = build_research_forest(simulation, scene, registry)
    scene_load_time = time.monotonic() - scene_started
    _, stage_snapshot = scene_paths(scene["scene_id"])
    export_stage_snapshot(stage_snapshot)
    scene_report = {
        "status": "PASS", "scene_id": scene["scene_id"], "scene_seed": scene["scene_seed"],
        "split": scene["split"], "scene_content_hash": scene["content_hash"], "factors": scene["factors"],
        "tree_count": len(scene["trees"]), "tree_asset_distribution": {
            asset_id: sum(tree["asset_id"] == asset_id for tree in scene["trees"])
            for asset_id in sorted({tree["asset_id"] for tree in scene["trees"]})
        },
        "terrain_statistics": built.terrain_statistics, "scene_load_time_s": scene_load_time,
        "missing_registry_assets": built.missing_registry_assets,
        "stage_snapshot": str(stage_snapshot.relative_to(ROOT)),
    }
    write_report(ARGS.output_directory / "scene_runtime_report.json", scene_report)

    if ARGS.plan_only:
        plan_results = []
        for episode_id in ARGS.episode_id:
            directory = ARGS.output_directory / episode_id
            directory.mkdir(parents=True, exist_ok=True)
            plan_path = directory / "episode_plan.yaml"
            if plan_path.exists():
                plan = load_yaml(plan_path)
                disposition = "REUSED_EXACT"
            else:
                plan = sample_episode_plan(scene, episode_id, built.surface.height)
                temporary_plan = directory / ".episode_plan.yaml.inprogress"
                dump_yaml(temporary_plan, plan)
                validate_plan(scene, plan)
                os.replace(temporary_plan, plan_path)
                disposition = "GENERATED"
            result = validate_plan(scene, plan)
            result["disposition"] = disposition
            result["plan_path"] = str(plan_path)
            plan_results.append(result)
        resources = monitor.stop()
        report = {
            "status": "PASS", "report_type": "plan_preflight_session", "scene_id": scene["scene_id"],
            "scene_content_hash": scene["content_hash"], "planner_version": PLANNER_VERSION,
            "app_startup_time_s": APP_READY - PROCESS_START, "scene_load_time_s": scene_load_time,
            "total_wall_time_s": time.monotonic() - PROCESS_START, "terrain_statistics": built.terrain_statistics,
            "plans": plan_results, "resources": resources,
        }
        write_report(ARGS.benchmark_output, report)
        print("MNS_DATASET_PLAN_PREFLIGHT_RESULT=" + json.dumps(report, sort_keys=True), flush=True)
        return 0

    if not ARGS.enable_cameras:
        raise ValueError("Dataset V0 collection requires --enable_cameras")
    missing_plans = [
        episode_id for episode_id in ARGS.episode_id
        if not (ARGS.output_directory / episode_id / "episode_plan.yaml").is_file()
    ]
    if missing_plans:
        raise FileNotFoundError(f"plan-only preflight must run before collection: {missing_plans}")

    surrogate = UsdGeom.Xform.Define(stage, "/World/DiabloSurrogate")
    surrogate.AddTranslateOp()
    surrogate.AddOrientOp()
    body = UsdGeom.Cube.Define(stage, "/World/DiabloSurrogate/Body")
    body.CreateSizeAttr(1.0)
    body.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.10, 0.13)])
    body.AddScaleOp().Set(Gf.Vec3f(float(robot["surrogate"]["footprint"]["length_m"]), float(robot["surrogate"]["footprint"]["width_m"]), float(robot["surrogate"]["body_height_m"])))
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(robot["surrogate"]["body_height_m"]) / 2.0))
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    semantic = Semantics.SemanticsAPI.Apply(body.GetPrim(), "Semantics")
    semantic.CreateSemanticTypeAttr().Set("class")
    semantic.CreateSemanticDataAttr().Set("robot")

    sensor_rate = float(sensor["application_profile"]["logging_rate_hz"])
    rgb_cfg, depth_cfg = sensor["application_profile"]["rgb"], sensor["application_profile"]["depth_raw"]
    rgb_camera = make_camera(sim_utils, Camera, CameraCfg, "/World/RgbCamera", rgb_cfg, sensor_rate, ["rgb"])
    depth_camera = make_camera(sim_utils, Camera, CameraCfg, "/World/DepthCamera", depth_cfg, sensor_rate, ["distance_to_image_plane"])
    UsdGeom.Camera(stage.GetPrimAtPath("/World/RgbCamera")).GetExposureAttr().Set(float(rgb_cfg["simulated_exposure_compensation_stops"]))
    simulation.reset()
    configured_rgb = np.asarray(camera_intrinsics(*rgb_cfg["resolution"], rgb_cfg["fov_deg"]["horizontal"], rgb_cfg["fov_deg"]["vertical"]), dtype=np.float64)
    configured_depth = np.asarray(camera_intrinsics(*depth_cfg["resolution"], depth_cfg["fov_deg"]["horizontal"], depth_cfg["fov_deg"]["vertical"]), dtype=np.float64)
    configured_rgb[1, 1], configured_depth[1, 1] = configured_rgb[0, 0], configured_depth[0, 0]
    rgb_intrinsics = rgb_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    depth_intrinsics = depth_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(rgb_intrinsics, configured_rgb, rtol=2.0e-3, atol=0.6) or not np.allclose(depth_intrinsics, configured_depth, rtol=2.0e-3, atol=0.6):
        raise RuntimeError("Isaac camera intrinsics differ from d435i_navigation_v0")
    print("MNS_DATASET_SESSION_READY=" + json.dumps({"scene_id": scene["scene_id"], "episodes": ARGS.episode_id, "scene_load_time_s": scene_load_time}), flush=True)

    results, endpoints = [], []
    for episode_id in ARGS.episode_id:
        start_index = len(monitor.samples)
        start_snapshot = monitor.snapshot()
        result = run_episode(
            simulation, stage, built, scene, robot, sensor, rgb_camera, depth_camera,
            rgb_intrinsics, depth_intrinsics, episode_id, ARGS.output_directory,
        )
        end_snapshot = monitor.snapshot()
        result["resources"] = monitor.summarize(start_index)
        endpoints.append({"episode_id": episode_id, "start": start_snapshot, "end": end_snapshot})
        results.append(result)
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    session_end_snapshot = monitor.snapshot()
    resources = monitor.stop()
    memory = _memory_leak_report(endpoints)
    rtf_values = [float(item["real_time_factor"]) for item in results if item["real_time_factor"] is not None]
    total_bytes = sum(int(item["hdf5_size_bytes"]) for item in results)
    report = {
        "status": "FAIL" if memory["suspected"] else "PASS",
        "report_type": "scene_collection_session", "report_version": 2,
        "scene_id": scene["scene_id"], "scene_content_hash": scene["content_hash"],
        "scene_schema_version": scene["schema_version"], "collector_version": COLLECTOR_VERSION,
        "episode_schema_version": EPISODE_SCHEMA_VERSION, "planner_version": PLANNER_VERSION,
        "session": {
            "app_startup_time_s": APP_READY - PROCESS_START, "scene_load_time_s": scene_load_time,
            "total_wall_time_s": time.monotonic() - PROCESS_START,
            "session_start_resources": session_start_snapshot, "session_end_resources": session_end_snapshot,
        },
        "episodes": results,
        "aggregate": {
            "all_pass": all(item["status"] in {"PASS", "SKIPPED_VALID"} for item in results) and not memory["suspected"],
            "episode_count": len(results), "total_simulated_time_s": sum(float(item["simulated_duration_s"]) for item in results),
            "total_episode_wall_time_s": sum(float(item["wall_execution_time_s"]) for item in results),
            "total_hdf5_bytes": total_bytes, "mean_rtf": float(np.mean(rtf_values)) if rtf_values else None,
            "minimum_rtf": min(rtf_values) if rtf_values else None,
        },
        "resources": resources, "memory_trend": memory, "terrain_statistics": built.terrain_statistics,
    }
    write_report(ARGS.benchmark_output, report)
    print("MNS_DATASET_RUNTIME_RESULT=" + json.dumps(report, sort_keys=True), flush=True)
    if memory["suspected"]:
        raise RuntimeError(f"scene batch resource growth failed stability gate: {memory['signals']}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        try:
            APP.close(wait_for_replicator=False, skip_cleanup=True)
        except SystemExit:
            # Some Kit shutdown paths raise SystemExit(0); never let that mask
            # a failed preflight or collection result.
            pass
    sys.exit(code)
