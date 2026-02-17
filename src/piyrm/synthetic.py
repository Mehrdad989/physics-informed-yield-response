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


def generate_synthetic_region_yield(
    *,
    seed: int = 0,
    n_regions: int = 8,
    n_per_region: int = 200,
    rain_min_mm: float = 0.0,
    rain_max_mm: float = 800.0,
    sigma_y: float = 0.4,
    include_true_params: bool = True,
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

    params = sample_region_params(rng=rng, n_regions=n_regions)

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
