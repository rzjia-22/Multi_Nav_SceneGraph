"""Project-owned NavDiffusion V0 network.

The conditional 1-D U-Net structure follows the ForestNavigation NavDiffusion
implementation audited at 0b29c399754f510499bfe9cc9d592cba215a7161.
That repository identifies itself as BSD-3-Clause but omits its root license at
the audited revision; the implementation here is rewritten and attribution is
retained in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as functional


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, maximum_length: int) -> None:
        super().__init__()
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisors = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        values = torch.zeros(maximum_length, dimension)
        values[:, 0::2] = torch.sin(positions * divisors)
        values[:, 1::2] = torch.cos(positions * divisors)
        self.register_buffer("values", values.unsqueeze(0), persistent=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.values[:, : value.shape[1]]


class GoalEncoder(nn.Module):
    def __init__(self, output_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dimension),
        )

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.layers(goal)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VisualGoalTransformer(nn.Module):
    def __init__(
        self,
        history_length: int = 5,
        embedding_dimension: int = 256,
        layers: int = 4,
        heads: int = 4,
        feed_forward_factor: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = efficientnet_b0(weights=weights)
        original_stem = encoder.features[0][0]
        rgb_weights = original_stem.weight.detach().clone()
        expanded_stem = nn.Conv2d(
            4,
            original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            dilation=original_stem.dilation,
            groups=original_stem.groups,
            bias=original_stem.bias is not None,
            padding_mode=original_stem.padding_mode,
        )
        with torch.no_grad():
            expanded_stem.weight.zero_()
            expanded_stem.weight[:, :3].copy_(rgb_weights)
            if expanded_stem.bias is not None:
                expanded_stem.bias.copy_(original_stem.bias)
        encoder.features[0][0] = expanded_stem
        self.encoder = encoder
        self.feature_dimension = int(encoder.classifier[1].in_features)
        self.compress = nn.Linear(self.feature_dimension, embedding_dimension)
        self.goal_encoder = GoalEncoder(embedding_dimension)
        self.position = PositionalEncoding(embedding_dimension, history_length + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=heads,
            dim_feedforward=feed_forward_factor * embedding_dimension,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        stem = self.encoder.features[0][0].weight.detach()
        self.initialization_report: dict[str, Any] = {
            "pretrained": pretrained,
            "rgb_stem_maximum_difference": float((stem[:, :3] - rgb_weights).abs().max()),
            "depth_stem_maximum_absolute_weight": float(stem[:, 3].abs().max()),
            "batch_norm_layer_count": sum(isinstance(item, nn.BatchNorm2d) for item in encoder.modules()),
            "normalization_replacement": "none",
            "weights": "IMAGENET1K_V1" if pretrained else "none",
            "weights_url": EfficientNet_B0_Weights.IMAGENET1K_V1.url,
        }
        if pretrained:
            checkpoint_name = Path(EfficientNet_B0_Weights.IMAGENET1K_V1.url).name
            checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
            if checkpoint_path.is_file():
                self.initialization_report["weights_sha256"] = _sha256(checkpoint_path)

    def forward(self, images: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        batch, history = images.shape[:2]
        flattened = images.reshape(batch * history, *images.shape[2:])
        features = self.encoder.features(flattened)
        features = self.encoder.avgpool(features).flatten(1)
        features = self.encoder.classifier[0](features)
        vision = self.compress(features).reshape(batch, history, -1)
        goal = self.goal_encoder(goals).unsqueeze(1)
        tokens = self.position(torch.cat((vision, goal), dim=1))
        return self.transformer(tokens).mean(dim=1)


class SinusoidalDiffusionEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        factor = math.log(10000.0) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32) * -factor
        )
        angles = timestep.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, output_channels),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dimension: int,
        conditional_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            (Conv1dBlock(input_channels, output_channels), Conv1dBlock(output_channels, output_channels))
        )
        condition_channels = output_channels * (2 if conditional_predict_scale else 1)
        self.condition = nn.Sequential(nn.Mish(), nn.Linear(condition_dimension, condition_channels))
        self.conditional_predict_scale = conditional_predict_scale
        self.output_channels = output_channels
        self.residual = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels else nn.Identity()
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        output = self.blocks[0](value)
        encoded = self.condition(condition).unsqueeze(-1)
        if self.conditional_predict_scale:
            scale, bias = encoded.reshape(encoded.shape[0], 2, self.output_channels, 1).unbind(1)
            output = scale * output + bias
        else:
            output = output + encoded
        return self.blocks[1](output) + self.residual(value)


class ConditionalUnet1d(nn.Module):
    def __init__(
        self,
        input_dimension: int = 2,
        global_condition_dimension: int = 256,
        down_dimensions: tuple[int, ...] = (64, 128, 256),
        conditional_predict_scale: bool = False,
        diffusion_embedding_dimension: int = 256,
    ) -> None:
        super().__init__()
        dimensions = (input_dimension, *down_dimensions)
        pairs = list(zip(dimensions[:-1], dimensions[1:]))
        self.diffusion_encoder = nn.Sequential(
            SinusoidalDiffusionEmbedding(diffusion_embedding_dimension),
            nn.Linear(diffusion_embedding_dimension, diffusion_embedding_dimension * 4),
            nn.Mish(),
            nn.Linear(diffusion_embedding_dimension * 4, diffusion_embedding_dimension),
        )
        condition_dimension = diffusion_embedding_dimension + global_condition_dimension
        block = lambda a, b: ConditionalResidualBlock1d(
            a, b, condition_dimension, conditional_predict_scale
        )
        self.down = nn.ModuleList()
        for index, (input_channels, output_channels) in enumerate(pairs):
            last = index == len(pairs) - 1
            self.down.append(nn.ModuleList((
                block(input_channels, output_channels),
                block(output_channels, output_channels),
                nn.Identity() if last else nn.Conv1d(output_channels, output_channels, 3, 2, 1),
            )))
        final_dimension = dimensions[-1]
        self.middle = nn.ModuleList((block(final_dimension, final_dimension), block(final_dimension, final_dimension)))
        self.up = nn.ModuleList()
        reverse_pairs = list(reversed(pairs[1:]))
        for input_channels, output_channels in reverse_pairs:
            self.up.append(nn.ModuleList((
                block(output_channels * 2, input_channels),
                block(input_channels, input_channels),
                nn.ConvTranspose1d(input_channels, input_channels, 4, 2, 1),
            )))
        self.final = nn.Sequential(
            Conv1dBlock(down_dimensions[0], down_dimensions[0]),
            nn.Conv1d(down_dimensions[0], input_dimension, 1),
        )

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor | int, global_condition: torch.Tensor
    ) -> torch.Tensor:
        value = sample.permute(0, 2, 1)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=value.device, dtype=torch.long)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(value.device).expand(value.shape[0])
        condition = torch.cat((self.diffusion_encoder(timestep), global_condition), dim=-1)
        skips = []
        for first, second, downsample in self.down:
            value = first(value, condition)
            value = second(value, condition)
            skips.append(value)
            value = downsample(value)
        for middle in self.middle:
            value = middle(value, condition)
        for first, second, upsample in self.up:
            value = torch.cat((value, skips.pop()), dim=1)
            value = upsample(second(first(value, condition), condition))
        return self.final(value).permute(0, 2, 1)


def squared_cosine_betas(steps: int, offset: float = 0.008) -> torch.Tensor:
    def alpha_bar(time: float) -> float:
        return math.cos((time + offset) / (1.0 + offset) * math.pi / 2.0) ** 2

    return torch.tensor(
        [min(1.0 - alpha_bar(index / steps) / alpha_bar((index - 1) / steps), 0.999)
         for index in range(1, steps + 1)],
        dtype=torch.float32,
    )


class NavDiffusionV0(nn.Module):
    def __init__(self, architecture: dict[str, Any], *, pretrained: bool = True) -> None:
        super().__init__()
        visual = architecture["visual_encoder"]
        transformer = architecture["transformer"]
        diffusion = architecture["diffusion"]
        self.trajectory_points = int(architecture["trajectory_points"])
        self.visual_transformer = VisualGoalTransformer(
            history_length=5,
            embedding_dimension=int(visual["embedding_dim"]),
            layers=int(transformer["layers"]),
            heads=int(transformer["heads"]),
            feed_forward_factor=int(transformer["feed_forward_factor"]),
            pretrained=pretrained,
        )
        self.noise_predictor = ConditionalUnet1d(
            input_dimension=int(diffusion["input_dimension"]),
            global_condition_dimension=int(visual["embedding_dim"]),
            down_dimensions=tuple(int(v) for v in diffusion["down_dimensions"]),
            conditional_predict_scale=bool(diffusion["conditional_predict_scale"]),
        )
        betas = squared_cosine_betas(int(diffusion["iterations"]))
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumulative", torch.cumprod(alphas, dim=0))

    @property
    def initialization_report(self) -> dict[str, Any]:
        return self.visual_transformer.initialization_report

    def condition(self, images: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return self.visual_transformer(images, goals)

    def add_noise(
        self, trajectory: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        cumulative = self.alphas_cumulative[timesteps].to(trajectory.dtype)
        return cumulative.sqrt()[:, None, None] * trajectory + (1.0 - cumulative).sqrt()[:, None, None] * noise

    def diffusion_loss(
        self,
        images: torch.Tensor,
        goals: torch.Tensor,
        normalized_trajectory: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch = images.shape[0]
        timesteps = torch.randint(
            len(self.betas), (batch,), device=images.device, generator=generator
        )
        noise = torch.randn(
            normalized_trajectory.shape,
            device=normalized_trajectory.device,
            dtype=normalized_trajectory.dtype,
            generator=generator,
        )
        noisy = self.add_noise(normalized_trajectory, noise, timesteps)
        predicted = self.noise_predictor(noisy, timesteps, self.condition(images, goals))
        return functional.mse_loss(predicted.float(), noise.float())

    @torch.no_grad()
    def sample_normalized(
        self,
        images: torch.Tensor,
        goals: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.condition(images, goals)
        sample = torch.randn(
            (images.shape[0], self.trajectory_points, 2),
            device=images.device,
            dtype=images.dtype,
            generator=generator,
        )
        for index in reversed(range(len(self.betas))):
            timestep = torch.full((images.shape[0],), index, device=images.device, dtype=torch.long)
            epsilon = self.noise_predictor(sample, timestep, condition)
            cumulative = self.alphas_cumulative[index].to(sample.dtype)
            previous = (
                self.alphas_cumulative[index - 1].to(sample.dtype)
                if index > 0 else torch.ones((), device=sample.device, dtype=sample.dtype)
            )
            current_alpha = cumulative / previous
            current_beta = 1.0 - current_alpha
            predicted_original = (
                sample - (1.0 - cumulative).sqrt() * epsilon
            ) / cumulative.sqrt()
            predicted_original = predicted_original.clamp(-1.0, 1.0)
            mean = (
                previous.sqrt() * current_beta / (1.0 - cumulative) * predicted_original
                + current_alpha.sqrt() * (1.0 - previous) / (1.0 - cumulative) * sample
            )
            if index > 0:
                variance = ((1.0 - previous) / (1.0 - cumulative) * current_beta).clamp_min(1.0e-20)
                noise = torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
                sample = mean + variance.sqrt() * noise
            else:
                sample = mean
        return sample


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
