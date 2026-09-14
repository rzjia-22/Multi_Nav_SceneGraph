"""Plain-PyTorch data preparation, sanity checks, training, and validation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .data import (
    PROJECT_ROOT, NavDiffusionWindowDataset, file_sha256, load_yaml, prepare_cache,
)
from .model import NavDiffusionV0, parameter_counts
from .preprocessing import NavDiffusionPreprocessor


PREPROCESSING_CONFIG = PROJECT_ROOT / "config/models/navigation_input_v0.yaml"
MODEL_CONFIG = PROJECT_ROOT / "config/models/navdiffusion_v0.yaml"


def _paths(model_config: dict[str, Any]) -> tuple[Path, Path]:
    run = PROJECT_ROOT / model_config["outputs"]["run_root"]
    formal = PROJECT_ROOT / model_config["outputs"]["formal_model_directory"]
    run.mkdir(parents=True, exist_ok=True)
    formal.mkdir(parents=True, exist_ok=True)
    return run, formal


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed(value: int) -> None:
    import random

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("NavDiffusion V0 training requires the configured CUDA GPU")
    return torch.device("cuda")


def _amp(device: torch.device, preference: str) -> tuple[str, torch.dtype | None]:
    if device.type != "cuda":
        return "disabled", None
    if preference == "bf16" and torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "fp16", torch.float16


def _autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


def _scaler(amp_name: str):
    """Scale fp16 gradients; bf16 has enough exponent range without scaling."""
    return torch.amp.GradScaler("cuda", enabled=amp_name == "fp16")


def _optimizer(model: NavDiffusionV0, config: dict[str, Any]):
    training = config["training"]
    encoder_parameters = list(model.visual_transformer.encoder.parameters())
    encoder_ids = {id(item) for item in encoder_parameters}
    other_parameters = [item for item in model.parameters() if id(item) not in encoder_ids]
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(training["encoder_learning_rate"]), "name": "imagenet_encoder"},
            {"params": other_parameters, "lr": float(training["other_learning_rate"]), "name": "transformer_goal_diffusion"},
        ],
        weight_decay=float(training["weight_decay"]),
    )


def _scheduler(optimizer, config: dict[str, Any]):
    maximum = int(config["training"]["maximum_epochs"])
    warmup = int(config["training"]["warmup_epochs"])

    def multiplier(epoch: int) -> float:
        completed = epoch + 1
        if completed <= warmup:
            return completed / warmup
        progress = min(max((completed - warmup) / max(maximum - warmup, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _move(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["images"].to(device, non_blocking=True),
        batch["goal"].to(device, non_blocking=True),
        batch["trajectory"].to(device, non_blocking=True),
    )


@torch.no_grad()
def _fixed_loss(model, loader, device, amp_dtype, seed: int) -> float:
    model.eval()
    losses = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        images, goals, trajectory = _move(batch, device)
        with _autocast(device, amp_dtype):
            loss = model.diffusion_loss(images, goals, trajectory, generator=generator)
        losses.append(float(loss))
    return float(np.mean(losses))


@torch.no_grad()
def _trajectory_metrics(model, loader, device, amp_dtype, seed: int, bound_m: float) -> dict[str, float]:
    model.eval()
    ade_sum = 0.0
    fde_sum = 0.0
    count = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.monotonic()
    for batch in loader:
        images, goals, target = _move(batch, device)
        with _autocast(device, amp_dtype):
            predicted = model.sample_normalized(images, goals, generator=generator)
        distance = torch.linalg.vector_norm((predicted.float() - target.float()) * bound_m, dim=-1)
        ade_sum += float(distance.mean(dim=1).sum())
        fde_sum += float(distance[:, -1].sum())
        count += int(distance.shape[0])
    return {
        "ADE_m": ade_sum / count,
        "FDE_m": fde_sum / count,
        "window_count": count,
        "wall_time_s": time.monotonic() - started,
        "seed": seed,
    }


def prepare_data(force: bool = False) -> dict[str, Any]:
    report = prepare_cache(PREPROCESSING_CONFIG, MODEL_CONFIG, force=force)
    config = load_yaml(MODEL_CONFIG)
    _, formal = _paths(config)
    preprocessor = NavDiffusionPreprocessor.from_files(
        PREPROCESSING_CONFIG, formal / "navdiffusion_v0_data_report.json"
    )
    (formal / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    resolved = {
        "source": load_yaml(PREPROCESSING_CONFIG),
        "resolved": preprocessor.to_dict(),
        "dataset_index_sha256": report["dataset_index_sha256"],
        "preprocessing_config_sha256": report["preprocessing_config_sha256"],
        "statistics_split": "train",
        "test_split_used": False,
    }
    (formal / "preprocessing.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def single_batch() -> dict[str, Any]:
    config = load_yaml(MODEL_CONFIG)
    training = config["training"]
    _, formal = _paths(config)
    _seed(int(training["seed"]))
    device = _device()
    amp_name, amp_dtype = _amp(device, str(training["amp_preference"]))
    dataset = NavDiffusionWindowDataset("train", PREPROCESSING_CONFIG, MODEL_CONFIG)
    attempted = [int(training["micro_batch_size"]), int(training["fallback_micro_batch_size"])]
    last_error = None
    for batch_size in attempted:
        try:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            batch = next(iter(loader))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = NavDiffusionV0(config["architecture"], pretrained=True).to(device)
            initialization = dict(model.initialization_report)
            if initialization["rgb_stem_maximum_difference"] != 0.0:
                raise AssertionError("ImageNet RGB stem weights changed during expansion")
            if initialization["depth_stem_maximum_absolute_weight"] != 0.0:
                raise AssertionError("depth stem weights are not exactly zero")
            if initialization["batch_norm_layer_count"] <= 0:
                raise AssertionError("pretrained EfficientNet BatchNorm was not preserved")
            optimizer = _optimizer(model, config)
            scaler = _scaler(amp_name)
            images, goals, trajectory = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp_dtype):
                loss = model.diffusion_loss(images, goals, trajectory)
            if not torch.isfinite(loss):
                raise AssertionError("single-batch loss is not finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.sqrt(sum(
                parameter.grad.detach().float().pow(2).sum()
                for parameter in model.parameters() if parameter.grad is not None
            )))
            if gradient_norm <= 0.0 or not math.isfinite(gradient_norm):
                raise AssertionError("single-batch gradients are invalid")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            report = {
                "status": "PASS",
                "batch_size": batch_size,
                "effective_batch_size": batch_size * (32 // batch_size),
                "amp_mode": amp_name,
                "loss": float(loss),
                "gradient_norm": gradient_norm,
                "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "input_shape": list(images.shape),
                "goal_shape": list(goals.shape),
                "trajectory_shape": list(trajectory.shape),
                "parameter_counts": parameter_counts(model),
                "efficientnet_initialization": initialization,
            }
            _write_json(formal / "sanity_single_batch.json", report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return report
        except torch.cuda.OutOfMemoryError as error:
            last_error = error
            if "model" in locals():
                del model
            torch.cuda.empty_cache()
    raise RuntimeError("single-batch sanity exhausted configured batch fallbacks") from last_error


def _overfit_fixed_loss(model, images, goals, trajectory, device, amp_dtype, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad(), _autocast(device, amp_dtype):
        return float(model.diffusion_loss(images, goals, trajectory, generator=generator))


def overfit_sanity() -> dict[str, Any]:
    config = load_yaml(MODEL_CONFIG)
    training = config["training"]
    sanity = config["sanity"]
    _, formal = _paths(config)
    batch_report = json.loads((formal / "sanity_single_batch.json").read_text(encoding="utf-8"))
    batch_size = min(int(batch_report["batch_size"]), int(sanity["overfit_windows"]))
    dataset = NavDiffusionWindowDataset(
        "train", PREPROCESSING_CONFIG, MODEL_CONFIG,
        episode_id=str(sanity["overfit_episode_id"]),
        maximum_windows=int(sanity["overfit_windows"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    _seed(int(training["seed"]))
    device = _device()
    amp_name, amp_dtype = _amp(device, str(training["amp_preference"]))
    model = NavDiffusionV0(config["architecture"], pretrained=True).to(device)
    optimizer = _optimizer(model, config)
    scaler = _scaler(amp_name)
    images, goals, trajectory = _move(batch, device)
    initial_loss = _overfit_fixed_loss(model, images, goals, trajectory, device, amp_dtype, 9001)
    model.eval()
    with torch.no_grad(), _autocast(device, amp_dtype):
        initial_prediction = model.sample_normalized(
            images, goals, generator=torch.Generator(device=device).manual_seed(9002)
        ).float()
    initial_ade = float(torch.linalg.vector_norm(
        (initial_prediction - trajectory.float()) * dataset.preprocessor.trajectory_bound_m, dim=-1
    ).mean())
    losses = []
    model.train()
    started = time.monotonic()
    for step in range(int(sanity["overfit_steps"])):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_dtype):
            loss = model.diffusion_loss(images, goals, trajectory)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss))
    final_loss = _overfit_fixed_loss(model, images, goals, trajectory, device, amp_dtype, 9001)
    model.eval()
    with torch.no_grad(), _autocast(device, amp_dtype):
        final_prediction = model.sample_normalized(
            images, goals, generator=torch.Generator(device=device).manual_seed(9002)
        ).float()
    final_ade = float(torch.linalg.vector_norm(
        (final_prediction - trajectory.float()) * dataset.preprocessor.trajectory_bound_m, dim=-1
    ).mean())
    temporary = PROJECT_ROOT / config["outputs"]["run_root"] / "overfit_state.pt"
    torch.save(model.state_dict(), temporary)
    reloaded = NavDiffusionV0(config["architecture"], pretrained=False).to(device)
    reloaded.load_state_dict(torch.load(temporary, map_location=device, weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad(), _autocast(device, amp_dtype):
        reloaded_prediction = reloaded.sample_normalized(
            images, goals, generator=torch.Generator(device=device).manual_seed(9002)
        ).float()
    reload_difference = float((final_prediction - reloaded_prediction).abs().max())
    required = float(sanity["required_loss_reduction_fraction"])
    reduction = 1.0 - final_loss / initial_loss
    report = {
        "status": "PASS" if reduction >= required and final_ade < initial_ade and reload_difference <= 1.0e-6 else "FAIL",
        "episode_id": sanity["overfit_episode_id"],
        "window_count": len(dataset),
        "batch_size": batch_size,
        "steps": int(sanity["overfit_steps"]),
        "amp_mode": amp_name,
        "initial_fixed_noise_loss": initial_loss,
        "final_fixed_noise_loss": final_loss,
        "loss_reduction_fraction": reduction,
        "initial_ADE_m": initial_ade,
        "final_ADE_m": final_ade,
        "checkpoint_reload_maximum_difference": reload_difference,
        "wall_time_s": time.monotonic() - started,
    }
    _write_json(formal / "sanity_overfit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError(f"small-overfit sanity failed: {report}")
    return report


def _checkpoint_payload(
    model, optimizer, scheduler, scaler, config, preprocessor, epoch, best_ade,
    best_fde, validation_loss, amp_mode, pretrained_initialization,
) -> dict[str, Any]:
    index_path = PROJECT_ROOT / load_yaml(PREPROCESSING_CONFIG)["raw_source"]["dataset_index"]
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    return {
        "checkpoint_version": int(config["checkpoint_version"]),
        "model_name": config["model_name"],
        "model_state_dict": model.state_dict(),
        "architecture_config": config["architecture"],
        "preprocessing_config": load_yaml(PREPROCESSING_CONFIG),
        "preprocessing": preprocessor.to_dict(),
        "image_size": [preprocessor.width, preprocessor.height],
        "history_length": preprocessor.history_length,
        "future_horizon": model.trajectory_points,
        "trajectory_scale": preprocessor.trajectory_bound_m,
        "rgb_mean": list(preprocessor.rgb_mean),
        "rgb_std": list(preprocessor.rgb_std),
        "depth_mean": preprocessor.depth_mean,
        "depth_std": preprocessor.depth_std,
        "depth_invalid_policy": "fill_far",
        "depth_max_range": preprocessor.depth_max_m,
        "dataset_version": "dataset_v0",
        "dataset_index_sha256": file_sha256(index_path),
        "git_commit": git_commit,
        "training_seed": int(config["training"]["seed"]),
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "gradient_scaler_state": scaler.state_dict(),
        "best_validation_ADE": best_ade,
        "validation_FDE": best_fde,
        "validation_diffusion_loss": validation_loss,
        "amp_mode": amp_mode,
        "efficientnet_pretrained": pretrained_initialization,
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train() -> dict[str, Any]:
    config = load_yaml(MODEL_CONFIG)
    training = config["training"]
    run, formal = _paths(config)
    single = json.loads((formal / "sanity_single_batch.json").read_text(encoding="utf-8"))
    overfit = json.loads((formal / "sanity_overfit.json").read_text(encoding="utf-8"))
    if single["status"] != "PASS" or overfit["status"] != "PASS":
        raise RuntimeError("both sanity gates must pass before full training")
    micro_batch = int(single["batch_size"])
    accumulation = 32 // micro_batch
    _seed(int(training["seed"]))
    device = _device()
    amp_name, amp_dtype = _amp(device, str(training["amp_preference"]))
    train_dataset = NavDiffusionWindowDataset("train", PREPROCESSING_CONFIG, MODEL_CONFIG)
    validation_dataset = NavDiffusionWindowDataset("validation", PREPROCESSING_CONFIG, MODEL_CONFIG)
    generator = torch.Generator().manual_seed(int(training["seed"]))
    train_loader = DataLoader(
        train_dataset, batch_size=micro_batch, shuffle=True,
        num_workers=int(training["workers"]), persistent_workers=int(training["workers"]) > 0,
        pin_memory=True, generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=micro_batch, shuffle=False,
        num_workers=max(1, int(training["workers"]) // 2), persistent_workers=True,
        pin_memory=True,
    )
    last_path = run / "last.pt"
    history_path = formal / "training_history.csv"
    start_epoch = 0
    best_ade = math.inf
    best_fde = math.inf
    best_epoch = -1
    if last_path.is_file():
        resume = torch.load(last_path, map_location="cpu", weights_only=True)
        model = NavDiffusionV0(config["architecture"], pretrained=False)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        start_epoch = int(resume["epoch"]) + 1
        best_ade = float(resume["best_validation_ADE"])
        best_fde = float(resume["validation_FDE"])
        best_epoch = int(resume.get("best_epoch", start_epoch - 1))
        pretrained_initialization = resume["efficientnet_pretrained"]
    else:
        model = NavDiffusionV0(config["architecture"], pretrained=True)
        pretrained_initialization = dict(model.initialization_report)
    model.to(device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    scaler = _scaler(amp_name)
    if last_path.is_file():
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume.get("gradient_scaler_state", {}))
    preprocessor = train_dataset.preprocessor
    header = [
        "epoch", "train_loss", "validation_loss", "validation_ADE_m", "validation_FDE_m",
        "encoder_lr", "other_lr", "epoch_wall_time_s", "peak_vram_mib", "checkpoint_status",
    ]
    if not history_path.is_file() or start_epoch == 0:
        history_path.write_text(",".join(header) + "\n", encoding="utf-8")
    maximum_epochs = int(training["maximum_epochs"])
    validation_interval = int(training["trajectory_validation_interval_epochs"])
    patience = int(training["early_stopping_patience_epochs"])
    overall_started = time.monotonic()
    overall_peak_vram_mib = 0.0
    stop_reason = "maximum_epochs"
    for epoch in range(start_epoch, maximum_epochs):
        epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch_index, batch in enumerate(train_loader):
            images, goals, trajectory = _move(batch, device)
            remaining_batches = len(train_loader) - batch_index
            group_divisor = min(accumulation, remaining_batches)
            with _autocast(device, amp_dtype):
                loss = model.diffusion_loss(images, goals, trajectory) / group_divisor
            scaler.scale(loss).backward()
            losses.append(float(loss) * group_divisor)
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        validation_loss = _fixed_loss(
            model, validation_loader, device, amp_dtype, int(training["validation_seed"])
        )
        metrics = None
        checkpoint_status = "last"
        if (epoch + 1) % validation_interval == 0:
            metrics = _trajectory_metrics(
                model, validation_loader, device, amp_dtype,
                int(training["validation_seed"]), preprocessor.trajectory_bound_m,
            )
            if metrics["ADE_m"] < best_ade:
                best_ade, best_fde, best_epoch = metrics["ADE_m"], metrics["FDE_m"], epoch
                checkpoint_status = "best"
        scheduler.step()
        payload = _checkpoint_payload(
            model, optimizer, scheduler, scaler, config, preprocessor, epoch,
            best_ade, best_fde, validation_loss, amp_name, pretrained_initialization,
        )
        payload["best_epoch"] = best_epoch
        _atomic_torch_save(payload, last_path)
        if checkpoint_status == "best":
            _atomic_torch_save(payload, formal / "best.pt")
            _write_json(formal / "validation_metrics.json", metrics)
        row = [
            epoch + 1, float(np.mean(losses)), validation_loss,
            "" if metrics is None else metrics["ADE_m"],
            "" if metrics is None else metrics["FDE_m"],
            optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"],
            time.monotonic() - epoch_started,
            torch.cuda.max_memory_allocated(device) / 2**20,
            checkpoint_status,
        ]
        overall_peak_vram_mib = max(overall_peak_vram_mib, float(row[8]))
        with history_path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(row)
        print(json.dumps({
            "epoch": epoch + 1, "train_loss": row[1], "validation_loss": validation_loss,
            "trajectory_metrics": metrics, "best_epoch": best_epoch + 1,
            "best_ADE_m": best_ade, "epoch_wall_time_s": row[7], "checkpoint": checkpoint_status,
        }, sort_keys=True), flush=True)
        if best_epoch >= 0 and epoch - best_epoch >= patience:
            stop_reason = "early_stopping_validation_ADE"
            break
    with history_path.open(encoding="utf-8", newline="") as stream:
        history_rows = list(csv.DictReader(stream))
    total_epoch_wall_time = sum(float(item["epoch_wall_time_s"]) for item in history_rows)
    best_validation_loss = min(float(item["validation_loss"]) for item in history_rows)
    summary = {
        "status": "PASS",
        "model_name": config["model_name"],
        "actual_micro_batch_size": micro_batch,
        "gradient_accumulation": accumulation,
        "effective_batch_size": micro_batch * accumulation,
        "amp_mode": amp_name,
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch + 1,
        "best_validation_ADE_m": best_ade,
        "best_validation_FDE_m": best_fde,
        "stop_reason": stop_reason,
        "latest_invocation_wall_time_s": time.monotonic() - overall_started,
        "total_epoch_wall_time_s": total_epoch_wall_time,
        "best_validation_diffusion_loss": best_validation_loss,
        "peak_vram_mib": max(
            overall_peak_vram_mib,
            max(float(item["peak_vram_mib"]) for item in history_rows),
        ),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "train_window_count": len(train_dataset),
        "validation_window_count": len(validation_dataset),
        "optimizer": "AdamW",
        "encoder_learning_rate": training["encoder_learning_rate"],
        "other_learning_rate": training["other_learning_rate"],
        "weight_decay": training["weight_decay"],
        "warmup_epochs": training["warmup_epochs"],
        "schedule": training["schedule"],
        "test_split_used": False,
        "checkpoint_sha256": file_sha256(formal / "best.pt"),
        "checkpoint_bytes": (formal / "best.pt").stat().st_size,
        "parameter_counts": parameter_counts(model),
    }
    _write_json(formal / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def inference_smoke() -> dict[str, Any]:
    import h5py
    from research_data.depth import align_depth_to_rgb
    from .predictor import MNSNavDiffusionConfig, MNSNavDiffusionPredictor

    config = load_yaml(MODEL_CONFIG)
    _, formal = _paths(config)
    checkpoint = formal / "best.pt"
    dataset = NavDiffusionWindowDataset("validation", PREPROCESSING_CONFIG, MODEL_CONFIG, maximum_windows=1)
    episode_id, local_index = dataset.records[0]
    arrays = dataset._episode_arrays(episode_id)
    anchor = int(arrays["anchor_indices"][local_index])
    index = json.loads((PROJECT_ROOT / "datasets/dataset_v0/dataset_index.json").read_text(encoding="utf-8"))
    entry = next(item for item in index["episodes"] if item["episode_id"] == episode_id)
    with h5py.File(PROJECT_ROOT / entry["relative_path"], "r") as source:
        start = anchor - dataset.history + 1
        z16 = source["sensors/depth_raw_z16"][start:anchor + 1]
        calibration = source["calibration"]
        aligned = align_depth_to_rgb(
            z16, float(calibration["depth_scale_m"][()]),
            calibration["depth_intrinsics"][:], calibration["rgb_intrinsics"][:],
            calibration["depth_to_rgb_translation_m"][:], calibration["depth_to_rgb_rotation_xyzw"][:],
            calibration["rgb_resolution_wh"][:],
        )
        rgb = source["sensors/rgb"][start:anchor + 1]
    runtime_tensor = dataset.preprocessor.prepare_history(rgb, aligned).numpy()
    training_tensor = dataset[0]["images"].numpy()
    consistency_difference = float(np.max(np.abs(runtime_tensor - training_tensor)))
    goal_local = np.asarray(arrays["goal_normalized"][local_index]) * dataset.preprocessor.goal_scale_m
    cuda = MNSNavDiffusionPredictor(MNSNavDiffusionConfig(checkpoint=checkpoint, device="cuda", output_waypoints=32))
    cuda_result = cuda(rgb.astype(np.float32) / 255.0, aligned[..., None], goal_local)
    cpu = MNSNavDiffusionPredictor(MNSNavDiffusionConfig(checkpoint=checkpoint, device="cpu", output_waypoints=32))
    cpu_result = cpu(rgb.astype(np.float32) / 255.0, aligned[..., None], goal_local)
    report = {
        "status": "PASS",
        "episode_id": episode_id,
        "training_runtime_preprocessing_maximum_difference": consistency_difference,
        "shape": list(cuda_result.shape),
        "cuda_finite": bool(np.isfinite(cuda_result).all()),
        "cpu_finite": bool(np.isfinite(cpu_result).all()),
        "cuda_inference_ms": cuda.last_inference_ms,
        "cpu_inference_ms": cpu.last_inference_ms,
        "maximum_absolute_waypoint_m": float(np.max(np.abs(cuda_result))),
        "checkpoint_sha256": file_sha256(checkpoint),
        "test_split_used": False,
    }
    if consistency_difference > 1.0e-6 or report["shape"] != [32, 2] or not report["cuda_finite"] or not report["cpu_finite"]:
        report["status"] = "FAIL"
    _write_json(formal / "inference_smoke.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError(f"inference smoke failed: {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--force", action="store_true")
    commands.add_parser("single-batch")
    commands.add_parser("overfit")
    commands.add_parser("train")
    commands.add_parser("inference-smoke")
    arguments = parser.parse_args()
    if arguments.command == "prepare-data":
        prepare_data(arguments.force)
    elif arguments.command == "single-batch":
        single_batch()
    elif arguments.command == "overfit":
        overfit_sanity()
    elif arguments.command == "train":
        train()
    elif arguments.command == "inference-smoke":
        inference_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
