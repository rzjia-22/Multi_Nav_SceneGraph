"""官方 NoMaD checkpoint 的严格、只推理适配器。"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_conditional_unet(runtime_hashes: dict[str, str]):
    """从 ML 镜像加载兼容实现，避开其顶层可选 Lightning 依赖。"""

    package_spec = importlib.util.find_spec("navdiffusion")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ModuleNotFoundError("the robotics ML image lacks navdiffusion")
    source_root = (
        Path(next(iter(package_spec.submodule_search_locations)))
        / "models" / "diffusion_policy"
    )
    for filename, expected_hash in runtime_hashes.items():
        path = source_root / filename
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"Conditional U-Net runtime source mismatch: {filename}")

    # navdiffusion.__init__ imports an optional Lightning wrapper. NoMaD only needs
    # the self-contained diffusion_policy package, so load it under a private name.
    package_name = "_mns_nomad_diffusion_policy"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_root)]
    sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.conditional_unet1d")
    return module.ConditionalUnet1D


class OfficialNoMaDAdapter:
    """加载官方源码/权重，并强制检查版本、形状和全部 state-dict 键。"""

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        import torch
        import yaml
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        self.torch = torch
        self.project_root = project_root
        self.config = config
        settings = config["nomad"]
        adapter = config["input_adapter"]
        source_root = project_root / settings["source_root"]
        checkpoint = project_root / settings["checkpoint"]
        official_config_path = project_root / settings["official_config"]
        if not source_root.is_dir() or not (source_root / ".git").exists():
            raise FileNotFoundError(
                f"official NoMaD source missing at {source_root}; run make nomad-assets"
            )
        revision = subprocess.check_output(
            [
                "git", "-c", f"safe.directory={source_root}",
                "-C", str(source_root), "rev-parse", "HEAD",
            ],
            text=True,
        ).strip()
        if revision != str(settings["source_revision"]):
            raise ValueError(f"NoMaD source revision mismatch: {revision}")
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"official NoMaD checkpoint missing at {checkpoint}; run make nomad-assets"
            )
        if checkpoint.stat().st_size != int(settings["checkpoint_size_bytes"]):
            raise ValueError("NoMaD checkpoint size mismatch")
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != str(settings["checkpoint_sha256"]):
            raise ValueError("NoMaD checkpoint SHA256 mismatch")

        with official_config_path.open("r", encoding="utf-8") as stream:
            official = yaml.safe_load(stream)
        expected = {
            "model_type": "nomad",
            "vision_encoder": "nomad_vint",
            "context_size": int(adapter["context_frames"]) - 1,
            "len_traj_pred": 8,
            "image_size": list(adapter["image_size_wh"]),
            "num_diffusion_iters": int(settings["diffusion_steps"]),
        }
        mismatches = {
            key: {"expected": value, "official": official.get(key)}
            for key, value in expected.items() if official.get(key) != value
        }
        if mismatches:
            raise ValueError(f"official NoMaD configuration mismatch: {mismatches}")

        train_root = source_root / "train"
        if str(train_root) not in sys.path:
            sys.path.insert(0, str(train_root))
        from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
        from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
        # 项目 ML 镜像中的实现来自 Stanford diffusion_policy，接口和权重键与
        # NoMaD 官方依赖一致；下面使用 strict=True 再做逐键验证。
        ConditionalUnet1D = load_conditional_unet(
            dict(settings["conditional_unet_source_sha256"])
        )

        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=int(official["encoding_size"]),
            context_size=int(official["context_size"]),
            mha_num_attention_heads=int(official["mha_num_attention_heads"]),
            mha_num_attention_layers=int(official["mha_num_attention_layers"]),
            mha_ff_dim_factor=int(official["mha_ff_dim_factor"]),
        )
        # 官方加载器会在恢复 checkpoint 前把 EfficientNet 的 BatchNorm 换成
        # GroupNorm；必须复现，否则模型结构与已发布权重并不相同。
        vision_encoder = replace_bn_with_gn(vision_encoder)
        noise_predictor = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=int(official["encoding_size"]),
            down_dims=list(official["down_dims"]),
            cond_predict_scale=bool(official["cond_predict_scale"]),
        )
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_predictor,
            dist_pred_net=DenseNetwork(int(official["encoding_size"])),
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_tensor_count = len(state_dict)
        state_value_count = sum(value.numel() for value in state_dict.values())
        if state_tensor_count != int(settings["expected_state_tensor_count"]):
            raise ValueError(f"unexpected NoMaD state tensor count: {state_tensor_count}")
        if state_value_count != int(settings["expected_state_value_count"]):
            raise ValueError(f"unexpected NoMaD state value count: {state_value_count}")
        model.load_state_dict(state_dict, strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != int(settings["expected_parameter_count"]):
            raise ValueError(f"unexpected NoMaD parameter count: {parameter_count}")

        requested_device = str(settings["device"])
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NoMaD formal evaluation requires the configured CUDA GPU")
        self.device = torch.device(requested_device)
        self.model = model.eval().to(self.device)
        self.official_config = official
        self.checkpoint_sha256 = checkpoint_sha
        self.source_revision = revision
        self.parameter_count = parameter_count
        self.state_tensor_count = state_tensor_count
        self.state_value_count = state_value_count
        self.scheduler = DDPMScheduler(
            num_train_timesteps=int(settings["diffusion_steps"]),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.delta_min = np.asarray(adapter["action_delta_min"], dtype=np.float32)
        self.delta_max = np.asarray(adapter["action_delta_max"], dtype=np.float32)
        self.image_size = tuple(int(value) for value in adapter["image_size_wh"])
        self.crop_ratio = float(adapter["center_crop_aspect_ratio"])
        self.rgb_mean = tuple(float(value) for value in adapter["rgb_mean"])
        self.rgb_std = tuple(float(value) for value in adapter["rgb_std"])
        self.trajectory_points = int(official["len_traj_pred"])

    def preprocess_images(self, images_rgb: np.ndarray):
        """复现训练端：4:3 中心裁剪、96×96、ImageNet 标准化。"""

        from PIL import Image
        from torchvision import transforms
        import torchvision.transforms.functional as vision_functional

        values = np.asarray(images_rgb)
        if values.ndim == 3:
            values = values[None]
        if values.ndim != 4 or values.shape[-1] != 3:
            raise ValueError("RGB images must have shape [N,H,W,3]")
        normalize = transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        tensors = []
        for value in values:
            image = Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB")
            width, height = image.size
            if width > height:
                image = vision_functional.center_crop(
                    image, (height, int(round(height * self.crop_ratio)))
                )
            else:
                image = vision_functional.center_crop(
                    image, (int(round(width / self.crop_ratio)), width)
                )
            image = image.resize(self.image_size)
            tensors.append(normalize(vision_functional.to_tensor(image)))
        return self.torch.stack(tensors).to(self.device)

    def encode(self, context_rgb: np.ndarray, goal_rgb: np.ndarray, *, goal_masked: bool):
        context = self.preprocess_images(context_rgb)
        if len(context) != int(self.config["input_adapter"]["context_frames"]):
            raise ValueError("NoMaD context frame count mismatch")
        context = context.reshape(1, -1, *context.shape[-2:])
        goals = self.preprocess_images(goal_rgb)
        context = context.repeat(len(goals), 1, 1, 1)
        mask_value = 1 if goal_masked else 0
        masks = self.torch.full(
            (len(goals),), mask_value, dtype=self.torch.long, device=self.device
        )
        with self.torch.inference_mode():
            condition = self.model(
                "vision_encoder", obs_img=context, goal_img=goals, input_goal_mask=masks
            )
        return condition

    def predict_distance(self, condition) -> np.ndarray:
        with self.torch.inference_mode():
            prediction = self.model("dist_pred_net", obsgoal_cond=condition).flatten()
        return prediction.float().cpu().numpy()

    def sample_trajectories(
        self,
        condition,
        *,
        sample_count: int,
        seed: int,
        action_scale_m: float,
    ) -> tuple[np.ndarray, float]:
        """生成以米表示的累计局部路点，形状为 [N,8,2]。"""

        torch = self.torch
        repeated = condition[:1].repeat(int(sample_count), 1)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        started = time.perf_counter()
        sample = torch.randn(
            (int(sample_count), self.trajectory_points, 2),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        self.scheduler.set_timesteps(int(self.config["nomad"]["diffusion_steps"]))
        with torch.inference_mode():
            for timestep in self.scheduler.timesteps:
                noise = self.model(
                    "noise_pred_net",
                    sample=sample,
                    timestep=timestep,
                    global_cond=repeated,
                )
                sample = self.scheduler.step(
                    model_output=noise,
                    timestep=timestep,
                    sample=sample,
                    generator=generator,
                ).prev_sample
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        normalized = sample.float().cpu().numpy()
        deltas = (normalized + 1.0) * 0.5 * (self.delta_max - self.delta_min) + self.delta_min
        trajectories = np.cumsum(deltas, axis=1) * float(action_scale_m)
        return trajectories.astype(np.float32), inference_ms
