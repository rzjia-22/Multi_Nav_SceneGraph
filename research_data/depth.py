"""Versioned RealSense-style depth storage and deterministic registration."""

from __future__ import annotations

from typing import Any

import numpy as np


ALIGNMENT_ALGORITHM = "z_buffer_reproject_depth_optical_to_rgb_optical"
ALIGNMENT_VERSION = 1
INVALID_DEPTH_CONVENTION = "zero_is_invalid"


def quantize_depth_z16(
    depth_m: np.ndarray,
    depth_scale_m: float,
    valid_range_m: tuple[float, float] | list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantize metric Isaac depth using the same value×scale contract as Z16."""
    source = np.asarray(depth_m, dtype=np.float32)
    minimum, maximum = (float(value) for value in valid_range_m)
    valid = np.isfinite(source) & (source >= minimum) & (source <= maximum)
    scaled = np.rint(np.where(valid, source, 0.0) / float(depth_scale_m))
    saturation = valid & (scaled > np.iinfo(np.uint16).max)
    encoded = np.clip(scaled, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    encoded[~valid] = 0
    restored = encoded.astype(np.float32) * float(depth_scale_m)
    errors = np.abs(restored[valid & ~saturation] - source[valid & ~saturation])
    report = {
        "depth_scale_m": float(depth_scale_m),
        "valid_range_m": [minimum, maximum],
        "invalid_convention": INVALID_DEPTH_CONVENTION,
        "input_valid_pixels": int(valid.sum()),
        "invalid_pixels": int((~valid).sum()),
        "saturation_count": int(saturation.sum()),
        "mean_absolute_error_m": float(errors.mean()) if errors.size else 0.0,
        "p95_absolute_error_m": float(np.percentile(errors, 95)) if errors.size else 0.0,
        "maximum_absolute_error_m": float(errors.max()) if errors.size else 0.0,
    }
    return encoded, report


def dequantize_depth_z16(depth_z16: np.ndarray, depth_scale_m: float) -> np.ndarray:
    encoded = np.asarray(depth_z16, dtype=np.uint16)
    depth = encoded.astype(np.float32) * float(depth_scale_m)
    depth[encoded == 0] = 0.0
    return depth


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        raise ValueError("zero-norm depth-to-RGB rotation")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def align_depth_to_rgb(
    depth_z16: np.ndarray,
    depth_scale_m: float,
    depth_intrinsics: np.ndarray,
    rgb_intrinsics: np.ndarray,
    depth_to_rgb_translation_m: np.ndarray,
    depth_to_rgb_rotation_xyzw: np.ndarray,
    rgb_resolution: tuple[int, int] | list[int],
) -> np.ndarray:
    """Register one or many raw Z16 frames into the RGB optical frame."""
    encoded = np.asarray(depth_z16, dtype=np.uint16)
    single = encoded.ndim == 2
    frames = encoded[None, ...] if single else encoded
    depth_k = np.asarray(depth_intrinsics, dtype=np.float64)
    rgb_k = np.asarray(rgb_intrinsics, dtype=np.float64)
    translation = np.asarray(depth_to_rgb_translation_m, dtype=np.float64)
    rotation = _quaternion_xyzw_to_matrix(depth_to_rgb_rotation_xyzw)
    output_width, output_height = (int(value) for value in rgb_resolution)
    vv, uu = np.indices(frames.shape[-2:], dtype=np.float64)
    outputs = []
    for frame in frames:
        z = dequantize_depth_z16(frame, depth_scale_m).astype(np.float64)
        valid = frame != 0
        points = np.stack(
            (
                (uu[valid] - depth_k[0, 2]) * z[valid] / depth_k[0, 0],
                (vv[valid] - depth_k[1, 2]) * z[valid] / depth_k[1, 1],
                z[valid],
            ),
            axis=0,
        )
        transformed = rotation @ points + translation[:, None]
        positive = transformed[2] > 0.0
        projected_u = np.rint(rgb_k[0, 0] * transformed[0, positive] / transformed[2, positive] + rgb_k[0, 2]).astype(np.int64)
        projected_v = np.rint(rgb_k[1, 1] * transformed[1, positive] / transformed[2, positive] + rgb_k[1, 2]).astype(np.int64)
        inside = (
            (projected_u >= 0)
            & (projected_u < output_width)
            & (projected_v >= 0)
            & (projected_v < output_height)
        )
        flat = np.full(output_height * output_width, np.inf, dtype=np.float32)
        np.minimum.at(
            flat,
            projected_v[inside] * output_width + projected_u[inside],
            transformed[2, positive][inside].astype(np.float32),
        )
        aligned = flat.reshape(output_height, output_width)
        aligned[~np.isfinite(aligned)] = 0.0
        outputs.append(aligned)
    result = np.stack(outputs)
    return result[0] if single else result


def alignment_round_trip_report(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if reference.shape != reconstructed.shape:
        raise ValueError(f"alignment shape mismatch: {reference.shape} != {reconstructed.shape}")
    reference_valid = reference > 0.0
    reconstructed_valid = reconstructed > 0.0
    agreement = reference_valid == reconstructed_valid
    common = reference_valid & reconstructed_valid
    difference = np.abs(reference[common] - reconstructed[common])
    return {
        "pixel_count": int(reference.size),
        "valid_pixel_agreement": float(agreement.mean()),
        "registration_coverage": float(reconstructed_valid.mean()),
        "common_valid_pixels": int(common.sum()),
        "mean_depth_difference_m": float(difference.mean()) if difference.size else 0.0,
        "maximum_depth_difference_m": float(difference.max()) if difference.size else 0.0,
    }
