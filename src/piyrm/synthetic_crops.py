from __future__ import annotations

import numpy as np

from piyrm.curves import yield_rain_saturating

CROP_NAMES = ("wheat", "maize")


def generate_synthetic_region_crop_yield(
    *,
    seed: int = 0,
    n_regions: int = 20,
    n_per_region_crop: int = 150,
    rain_min_mm: float = 0.0,
    rain_max_mm: float = 800.0,
    sigma_y: float = 0.7,
    y_min: float = 0.5,
    # If provided, params will be driven by these features (recommended)
    region_features: dict[str, np.ndarray] | None = None,
    # crop effects
    crop_ymax_offsets: tuple[float, float] = (-0.3, 0.8),   # (wheat, maize)
    crop_k_multipliers: tuple[float, float] = (0.90, 1.10), # (wheat, maize)
    # small parameter noise
    param_noise_scale: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Synthetic dataset for cross-crop + cross-region with saturating curve.

    Each region has both crops. Each (crop, region) has its own (ymax, k).

    If region_features is provided, base region parameters are a function of features,
    so ML can learn features -> parameters.
    """
    rng = np.random.default_rng(seed)

    n_crops = 2
    n = n_regions * n_crops * n_per_region_crop

    # indices
    region = np.repeat(np.arange(n_regions, dtype=int), n_crops * n_per_region_crop)
    crop_id = np.tile(np.repeat(np.arange(n_crops, dtype=int), n_per_region_crop), n_regions)

    # rainfall observations
    rain_mm = rng.uniform(rain_min_mm, rain_max_mm, size=n)

    # ----- base region parameters -----
    if region_features is None:
        # random (no ML signal) - ok for pure curve demos, not for ML
        ymax_base = rng.uniform(5.5, 9.5, size=n_regions)
        k_base = rng.uniform(0.0015, 0.0080, size=n_regions)
    else:
        # feature-driven (ML signal)
        reg = np.asarray(region_features["region"], dtype=int)
        if reg.min() != 0 or reg.max() != n_regions - 1:
            raise ValueError("region_features['region'] must be exactly 0..n_regions-1")

        aridity = np.asarray(region_features["aridity_index"], dtype=float)
        soil = np.asarray(region_features["soil_whc"], dtype=float)
        heat = np.asarray(region_features["heat_stress"], dtype=float)

        soil_n = (soil - 50.0) / (250.0 - 50.0)

        # Base mappings (same spirit as your single-crop version)
        ymax_base = 6.5 + 2.0 * soil_n + 1.0 * aridity - 2.0 * heat
        k_base = 0.0035 + 0.0025 * aridity + 0.0015 * soil_n - 0.0020 * heat

        # small noise in base params
        ymax_base += rng.normal(0.0, param_noise_scale, size=n_regions)
        k_base += rng.normal(0.0, 0.0010 * param_noise_scale, size=n_regions)

        ymax_base = np.clip(ymax_base, 5.0, 10.0)
        k_base = np.clip(k_base, 0.0010, 0.0100)

    # ----- apply crop effects to get per-observation true params -----
    ymax_true = np.empty(n, dtype=float)
    k_true = np.empty(n, dtype=float)

    for r in range(n_regions):
        for c in range(n_crops):
            mask = (region == r) & (crop_id == c)
            ymax_true[mask] = ymax_base[r] + crop_ymax_offsets[c]
            k_true[mask] = k_base[r] * crop_k_multipliers[c]

    ymax_true = np.clip(ymax_true, 4.0, 11.0)
    k_true = np.clip(k_true, 0.0010, 0.0120)

    yield_true = yield_rain_saturating(rain_mm=rain_mm, ymax=ymax_true, k=k_true, y_min=y_min)

    yield_obs = yield_true + rng.normal(0.0, sigma_y, size=n)
    yield_obs = np.maximum(yield_obs, y_min)

    crop = np.array([CROP_NAMES[i] for i in crop_id], dtype=object)

    return {
        "region": region,
        "crop_id": crop_id,
        "crop": crop,
        "rain_mm": rain_mm,
        "yield_true": yield_true,
        "yield_obs": yield_obs,
        "y_min_true": np.full(n, y_min, dtype=float),
        "ymax_true": ymax_true,
        "k_true": k_true,
    }