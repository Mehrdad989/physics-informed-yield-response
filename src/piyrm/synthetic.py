from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from piyrm.curves import yield_rain_hump


@dataclass(frozen=True)
class RegionParams:
    region: int
    ymax: float
    r_opt_mm: float
    width_mm: float
    y_min: float


def sample_region_params(
    rng: np.random.Generator,
    n_regions: int,
    ymax_range: tuple[float, float] = (5.0, 10.0),
    r_opt_range_mm: tuple[float, float] = (300.0, 600.0),
    width_range_mm: tuple[float, float] = (80.0, 200.0),
    y_min: float = 0.5,
) -> list[RegionParams]:
    """Sample per-region curve parameters from simple uniform ranges."""
    if n_regions <= 0:
        raise ValueError("n_regions must be > 0")

    ymax_lo, ymax_hi = ymax_range
    ropt_lo, ropt_hi = r_opt_range_mm
    w_lo, w_hi = width_range_mm

    if not (0 <= y_min):
        raise ValueError("y_min must be >= 0")
    if not (ymax_hi >= ymax_lo >= y_min):
        raise ValueError("ymax_range must satisfy ymax_hi >= ymax_lo >= y_min")
    if not (ropt_hi >= ropt_lo >= 0):
        raise ValueError("r_opt_range_mm must be >= 0 and hi >= lo")
    if not (w_hi > 0 and w_hi >= w_lo > 0):
        raise ValueError("width_range_mm must be > 0 and hi >= lo")

    ymax = rng.uniform(ymax_lo, ymax_hi, size=n_regions)
    r_opt = rng.uniform(ropt_lo, ropt_hi, size=n_regions)
    width = rng.uniform(w_lo, w_hi, size=n_regions)

    return [
        RegionParams(region=i, ymax=float(ymax[i]), r_opt_mm=float(r_opt[i]), width_mm=float(width[i]), y_min=float(y_min))
        for i in range(n_regions)
    ]

def region_params_from_features(
    *,
    rng: np.random.Generator,
    features: dict[str, np.ndarray],
    y_min: float = 0.5,
    noise_scale: float = 0.15,
) -> list[RegionParams]:
    """
    Create region parameters from region-level features + small noise.

    features must include arrays of equal length:
      - region (int)
      - aridity_index in [0,1]
      - soil_whc in [50,250]
      - heat_stress in [0,1]
    """
    region = np.asarray(features["region"], dtype=int)
    aridity = np.asarray(features["aridity_index"], dtype=float)
    soil = np.asarray(features["soil_whc"], dtype=float)
    heat = np.asarray(features["heat_stress"], dtype=float)

    n = region.size
    if not (aridity.size == soil.size == heat.size == n):
        raise ValueError("All feature arrays must have the same length")

    # Normalize soil roughly to 0..1
    soil_n = (soil - 50.0) / (250.0 - 50.0)

    # Base + effects (chosen to keep values in realistic ranges after clipping)
    ymax = 6.5 + 2.0 * soil_n + 1.0 * aridity - 2.0 * heat
    r_opt = 350.0 + 180.0 * aridity + 40.0 * soil_n - 60.0 * heat
    width = 100.0 + 80.0 * soil_n - 30.0 * heat

    # Add small parameter noise
    ymax += rng.normal(0.0, noise_scale, size=n)
    r_opt += rng.normal(0.0, 20.0 * noise_scale, size=n)
    width += rng.normal(0.0, 40.0 * noise_scale, size=n)

    # Clip into allowed ranges (physics-informed bounds)
    ymax = np.clip(ymax, 5.0, 10.0)
    r_opt = np.clip(r_opt, 300.0, 600.0)
    width = np.clip(width, 80.0, 200.0)

    return [
        RegionParams(
            region=int(region[i]),
            ymax=float(ymax[i]),
            r_opt_mm=float(r_opt[i]),
            width_mm=float(width[i]),
            y_min=float(y_min),
        )
        for i in range(n)
    ]

def generate_synthetic_region_yield(
    *,
    seed: int = 0,
    n_regions: int = 8,
    n_per_region: int = 200,
    rain_min_mm: float = 0.0,
    rain_max_mm: float = 800.0,
    sigma_y: float = 0.4,
    include_true_params: bool = True,
    region_features: dict[str, np.ndarray] | None = None,
    param_noise_scale: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Generate a synthetic dataset with region-specific yield response parameters.

    Returns a dict-of-arrays (easy to later convert to pandas if desired) with:
      - region, rain_mm, yield_true, yield_obs
      - optionally: ymax_true, r_opt_true, width_true, y_min_true
    """
    if n_per_region <= 0:
        raise ValueError("n_per_region must be > 0")
    if rain_max_mm < rain_min_mm:
        raise ValueError("rain_max_mm must be >= rain_min_mm")
    if sigma_y < 0:
        raise ValueError("sigma_y must be >= 0")

    rng = np.random.default_rng(seed)

    if region_features is None:
        params = sample_region_params(rng=rng, n_regions=n_regions)
    else:
        if int(np.max(region_features["region"])) != n_regions - 1:
            raise ValueError("region_features['region'] must be 0..n_regions-1")
        params = region_params_from_features(
            rng=rng,
            features=region_features,
            noise_scale=param_noise_scale,
        )

    n = n_regions * n_per_region
    region = np.repeat(np.arange(n_regions, dtype=int), n_per_region)

    rain_mm = rng.uniform(rain_min_mm, rain_max_mm, size=n)

    # Compute true yield per row using the region's parameters
    yield_true = np.empty(n, dtype=float)
    y_min_true = np.empty(n, dtype=float)
    ymax_true = np.empty(n, dtype=float)
    r_opt_true = np.empty(n, dtype=float)
    width_true = np.empty(n, dtype=float)

    for r in range(n_regions):
        mask = region == r
        p = params[r]
        y_min_true[mask] = p.y_min
        ymax_true[mask] = p.ymax
        r_opt_true[mask] = p.r_opt_mm
        width_true[mask] = p.width_mm
        yield_true[mask] = yield_rain_hump(
            rain_mm=rain_mm[mask],
            ymax=p.ymax,
            r_opt_mm=p.r_opt_mm,
            width_mm=p.width_mm,
            y_min=p.y_min,
        )

    noise = rng.normal(loc=0.0, scale=sigma_y, size=n)
    yield_obs = yield_true + noise

    # Enforce non-negative / floor constraint
    yield_obs = np.maximum(yield_obs, y_min_true)

    out: dict[str, np.ndarray] = {
        "region": region,
        "rain_mm": rain_mm,
        "yield_true": yield_true,
        "yield_obs": yield_obs,
    }

    if include_true_params:
        out.update(
            {
                "y_min_true": y_min_true,
                "ymax_true": ymax_true,
                "r_opt_true": r_opt_true,
                "width_true": width_true,
            }
        )

    return out
