from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piyrm.curves import yield_rain_hump, yield_rain_saturating


@dataclass(frozen=True)
class RegionParams:
    region: int
    ymax: float
    y_min: float

    # hump params
    r_opt_mm: float | None = None
    width_mm: float | None = None

    # saturating param
    k: float | None = None


def sample_region_params_hump(
    rng: np.random.Generator,
    n_regions: int,
    ymax_range: tuple[float, float] = (5.0, 10.0),
    r_opt_range_mm: tuple[float, float] = (300.0, 600.0),
    width_range_mm: tuple[float, float] = (80.0, 200.0),
    y_min: float = 0.5,
) -> list[RegionParams]:
    """Sample per-region hump-curve parameters from simple uniform ranges."""
    if n_regions <= 0:
        raise ValueError("n_regions must be > 0")

    ymax_lo, ymax_hi = ymax_range
    ropt_lo, ropt_hi = r_opt_range_mm
    w_lo, w_hi = width_range_mm

    if y_min < 0:
        raise ValueError("y_min must be >= 0")
    if ymax_hi < ymax_lo or ymax_lo < y_min:
        raise ValueError("ymax_range must satisfy ymax_hi >= ymax_lo >= y_min")
    if ropt_lo < 0 or ropt_hi < ropt_lo:
        raise ValueError("r_opt_range_mm must satisfy 0 <= lo <= hi")
    if w_lo <= 0 or w_hi < w_lo:
        raise ValueError("width_range_mm must satisfy 0 < lo <= hi")

    ymax = rng.uniform(ymax_lo, ymax_hi, size=n_regions)
    r_opt = rng.uniform(ropt_lo, ropt_hi, size=n_regions)
    width = rng.uniform(w_lo, w_hi, size=n_regions)

    return [
        RegionParams(
            region=i,
            ymax=float(ymax[i]),
            y_min=float(y_min),
            r_opt_mm=float(r_opt[i]),
            width_mm=float(width[i]),
        )
        for i in range(n_regions)
    ]


def sample_region_params_saturating(
    rng: np.random.Generator,
    n_regions: int,
    ymax_range: tuple[float, float] = (5.0, 10.0),
    k_range: tuple[float, float] = (0.0015, 0.0080),
    y_min: float = 0.5,
) -> list[RegionParams]:
    """Sample per-region saturating-curve parameters from simple uniform ranges."""
    if n_regions <= 0:
        raise ValueError("n_regions must be > 0")

    ymax_lo, ymax_hi = ymax_range
    k_lo, k_hi = k_range

    if y_min < 0:
        raise ValueError("y_min must be >= 0")
    if ymax_hi < ymax_lo or ymax_lo < y_min:
        raise ValueError("ymax_range must satisfy ymax_hi >= ymax_lo >= y_min")
    if k_lo <= 0 or k_hi < k_lo:
        raise ValueError("k_range must satisfy 0 < lo <= hi")

    ymax = rng.uniform(ymax_lo, ymax_hi, size=n_regions)
    k = rng.uniform(k_lo, k_hi, size=n_regions)

    return [
        RegionParams(
            region=i,
            ymax=float(ymax[i]),
            y_min=float(y_min),
            k=float(k[i]),
        )
        for i in range(n_regions)
    ]


def region_params_from_features_hump(
    *,
    rng: np.random.Generator,
    features: dict[str, np.ndarray],
    y_min: float = 0.5,
    noise_scale: float = 0.15,
) -> list[RegionParams]:
    """
    Hump params from region-level features + small noise.
    Produces: ymax, r_opt_mm, width_mm.
    """
    region = np.asarray(features["region"], dtype=int)
    aridity = np.asarray(features["aridity_index"], dtype=float)
    soil = np.asarray(features["soil_whc"], dtype=float)
    heat = np.asarray(features["heat_stress"], dtype=float)

    n = region.size
    if not (aridity.size == soil.size == heat.size == n):
        raise ValueError("All feature arrays must have the same length")

    soil_n = (soil - 50.0) / (250.0 - 50.0)

    ymax = 6.5 + 2.0 * soil_n + 1.0 * aridity - 2.0 * heat
    r_opt = 350.0 + 180.0 * aridity + 40.0 * soil_n - 60.0 * heat
    width = 100.0 + 80.0 * soil_n - 30.0 * heat

    ymax += rng.normal(0.0, noise_scale, size=n)
    r_opt += rng.normal(0.0, 20.0 * noise_scale, size=n)
    width += rng.normal(0.0, 40.0 * noise_scale, size=n)

    ymax = np.clip(ymax, 5.0, 10.0)
    r_opt = np.clip(r_opt, 300.0, 600.0)
    width = np.clip(width, 80.0, 200.0)

    return [
        RegionParams(
            region=int(region[i]),
            ymax=float(ymax[i]),
            y_min=float(y_min),
            r_opt_mm=float(r_opt[i]),
            width_mm=float(width[i]),
        )
        for i in range(n)
    ]


def region_params_from_features_saturating(
    *,
    rng: np.random.Generator,
    features: dict[str, np.ndarray],
    y_min: float = 0.5,
    noise_scale: float = 0.15,
) -> list[RegionParams]:
    """
    Saturating params from region-level features + small noise.
    Produces: ymax, k.
    """
    region = np.asarray(features["region"], dtype=int)
    aridity = np.asarray(features["aridity_index"], dtype=float)
    soil = np.asarray(features["soil_whc"], dtype=float)
    heat = np.asarray(features["heat_stress"], dtype=float)

    n = region.size
    if not (aridity.size == soil.size == heat.size == n):
        raise ValueError("All feature arrays must have the same length")

    soil_n = (soil - 50.0) / (250.0 - 50.0)

    # ymax similar idea
    ymax = 6.5 + 2.0 * soil_n + 1.0 * aridity - 2.0 * heat

    # k controls how quickly we approach ymax.
    # More arid / more heat -> smaller k (slower saturation)
    k = 0.0035 + 0.0025 * aridity + 0.0015 * soil_n - 0.0020 * heat

    ymax += rng.normal(0.0, noise_scale, size=n)
    k += rng.normal(0.0, 0.0010 * noise_scale, size=n)

    ymax = np.clip(ymax, 5.0, 10.0)
    k = np.clip(k, 0.0010, 0.0100)

    return [
        RegionParams(
            region=int(region[i]),
            ymax=float(ymax[i]),
            y_min=float(y_min),
            k=float(k[i]),
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
    curve_family: str = "hump",  # "hump" or "saturating"
) -> dict[str, np.ndarray]:
    """
    Generate a synthetic dataset with region-specific yield response parameters.

    Returns a dict-of-arrays with:
      - region, rain_mm, yield_true, yield_obs
      - plus true parameters depending on curve_family
    """
    if n_per_region <= 0:
        raise ValueError("n_per_region must be > 0")
    if rain_max_mm < rain_min_mm:
        raise ValueError("rain_max_mm must be >= rain_min_mm")
    if sigma_y < 0:
        raise ValueError("sigma_y must be >= 0")
    if curve_family not in {"hump", "saturating"}:
        raise ValueError("curve_family must be 'hump' or 'saturating'")

    rng = np.random.default_rng(seed)

    # Build region params
    if region_features is None:
        if curve_family == "hump":
            params = sample_region_params_hump(rng=rng, n_regions=n_regions)
        else:
            params = sample_region_params_saturating(rng=rng, n_regions=n_regions)
    else:
        # validate region ids
        reg = np.asarray(region_features["region"], dtype=int)
        if reg.min() != 0 or reg.max() != n_regions - 1:
            raise ValueError("region_features['region'] must be exactly 0..n_regions-1")
        if curve_family == "hump":
            params = region_params_from_features_hump(rng=rng, features=region_features, noise_scale=param_noise_scale)
        else:
            params = region_params_from_features_saturating(rng=rng, features=region_features, noise_scale=param_noise_scale)

    n = n_regions * n_per_region
    region = np.repeat(np.arange(n_regions, dtype=int), n_per_region)

    rain_mm = rng.uniform(rain_min_mm, rain_max_mm, size=n)

    yield_true = np.empty(n, dtype=float)

    # store per-row true params (for debugging/validation)
    y_min_true = np.empty(n, dtype=float)
    ymax_true = np.empty(n, dtype=float)

    # hump-only
    r_opt_true = np.full(n, np.nan, dtype=float)
    width_true = np.full(n, np.nan, dtype=float)

    # saturating-only
    k_true = np.full(n, np.nan, dtype=float)

    for r in range(n_regions):
        mask = region == r
        p = params[r]

        y_min_true[mask] = p.y_min
        ymax_true[mask] = p.ymax

        if curve_family == "hump":
            if p.r_opt_mm is None or p.width_mm is None:
                raise ValueError("Hump curve requires r_opt_mm and width_mm in RegionParams.")
            r_opt_true[mask] = p.r_opt_mm
            width_true[mask] = p.width_mm
            yield_true[mask] = yield_rain_hump(
                rain_mm=rain_mm[mask],
                ymax=p.ymax,
                r_opt_mm=p.r_opt_mm,
                width_mm=p.width_mm,
                y_min=p.y_min,
            )
        else:
            if p.k is None:
                raise ValueError("Saturating curve requires k in RegionParams.")
            k_true[mask] = p.k
            yield_true[mask] = yield_rain_saturating(
                rain_mm=rain_mm[mask],
                ymax=p.ymax,
                k=p.k,
                y_min=p.y_min,
            )

    noise = rng.normal(loc=0.0, scale=sigma_y, size=n)
    yield_obs = yield_true + noise
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
            }
        )
        if curve_family == "hump":
            out.update({"r_opt_true": r_opt_true, "width_true": width_true})
        else:
            out.update({"k_true": k_true})

    return out