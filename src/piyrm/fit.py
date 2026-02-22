from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit

from piyrm.curves import yield_rain_hump


@dataclass(frozen=True)
class FitResult:
    region: int
    ymax: float
    r_opt_mm: float
    width_mm: float
    y_min: float

@dataclass(frozen=True)
class FitSaturatingResult:
    region: int
    ymax: float
    k: float
    y_min: float

def _model(rain_mm: np.ndarray, ymax: float, r_opt_mm: float, width_mm: float, y_min: float) -> np.ndarray:
    return yield_rain_hump(
        rain_mm=np.asarray(rain_mm),
        ymax=float(ymax),
        r_opt_mm=float(r_opt_mm),
        width_mm=float(width_mm),
        y_min=float(y_min),
    )
def _model_saturating_reparam(rain_mm: np.ndarray, y_min: float, delta: float, k: float) -> np.ndarray:
    from piyrm.curves import yield_rain_saturating

    ymax = float(y_min) + float(delta)
    return yield_rain_saturating(
        rain_mm=np.asarray(rain_mm),
        ymax=ymax,
        k=float(k),
        y_min=float(y_min),
    )    

def _model_reparam(rain_mm: np.ndarray, y_min: float, delta: float, r_opt_mm: float, width_mm: float) -> np.ndarray:
    ymax = float(y_min) + float(delta)
    return yield_rain_hump(
        rain_mm=np.asarray(rain_mm),
        ymax=ymax,
        r_opt_mm=float(r_opt_mm),
        width_mm=float(width_mm),
        y_min=float(y_min),
    )

def fit_saturating_by_region(
    *,
    region: np.ndarray,
    rain_mm: np.ndarray,
    yield_obs: np.ndarray,
    regions: Iterable[int] | None = None,
    p0_k: float = 0.004,
) -> list[FitSaturatingResult]:
    """
    Fit monotonic saturating curve parameters independently for each region.

    Curve:
      y = y_min + (ymax - y_min) * (1 - exp(-k * rain))
    """
    region = np.asarray(region, dtype=int)
    rain_mm = np.asarray(rain_mm, dtype=float)
    yield_obs = np.asarray(yield_obs, dtype=float)

    if not (region.shape == rain_mm.shape == yield_obs.shape):
        raise ValueError("region, rain_mm, yield_obs must have the same shape")
    if region.ndim != 1:
        raise ValueError("inputs must be 1D arrays")

    if regions is None:
        regions = np.unique(region)

    results: list[FitSaturatingResult] = []

    # bounds for (y_min, delta, k)
    bounds = (
        [0.0, 0.0, 1e-6],   # lower
        [50.0, 50.0, 1.0],  # upper (k=1 is extremely fast saturation, but ok)
    )

    for r in regions:
        mask = region == int(r)
        x = rain_mm[mask]
        y = yield_obs[mask]

        if x.size < 10:
            raise ValueError(f"Region {r} has too few observations ({x.size}). Need at least 10.")

        ymin0 = float(np.percentile(y, 5))
        ymax0 = float(np.percentile(y, 95))
        delta0 = max(1e-3, ymax0 - ymin0)

        p0 = [ymin0, delta0, float(p0_k)]

        popt, _ = curve_fit(_model_saturating_reparam, x, y, p0=p0, bounds=bounds, maxfev=30000)
        y_min_hat, delta_hat, k_hat = popt
        ymax_hat = float(y_min_hat) + float(delta_hat)

        results.append(
            FitSaturatingResult(
                region=int(r),
                ymax=float(ymax_hat),
                k=float(k_hat),
                y_min=float(y_min_hat),
            )
        )

    return results

def fit_hump_by_region(
    *,
    region: np.ndarray,
    rain_mm: np.ndarray,
    yield_obs: np.ndarray,
    regions: Iterable[int] | None = None,
    p0_width_mm: float = 150.0,
) -> list[FitResult]:
    """
    Fit hump-shaped curve parameters independently for each region.

    Inputs are 1D arrays of equal length:
      - region: integer region id per observation
      - rain_mm: seasonal rainfall (mm)
      - yield_obs: observed yield

    Returns: list of FitResult (one per region)
    """
    region = np.asarray(region, dtype=int)
    rain_mm = np.asarray(rain_mm, dtype=float)
    yield_obs = np.asarray(yield_obs, dtype=float)

    if not (region.shape == rain_mm.shape == yield_obs.shape):
        raise ValueError("region, rain_mm, yield_obs must have the same shape")
    if region.ndim != 1:
        raise ValueError("inputs must be 1D arrays")

    if regions is None:
        regions = np.unique(region)

    results: list[FitResult] = []

    bounds = (
        [0.0, 0.0, 1.0, 0.0],         # lower: ymax, r_opt, width, y_min
        [50.0, 2000.0, 2000.0, 50.0], # upper
    )

    for r in regions:
        mask = region == int(r)
        x = rain_mm[mask]
        y = yield_obs[mask]

        if x.size < 10:
            raise ValueError(f"Region {r} has too few observations ({x.size}). Need at least 10.")

        ymin0 = float(np.percentile(y, 5))
        ymax0 = float(np.percentile(y, 95))
        delta0 = max(1e-3, ymax0 - ymin0)

        p0 = [
            ymin0,                 # y_min
            delta0,                # delta = ymax - y_min
            float(np.median(x)),   # r_opt
            float(p0_width_mm),    # width
        ]

        # Bounds for (y_min, delta, r_opt, width)
        bounds = (
            [0.0, 0.0, 0.0, 1.0],          # lower
            [50.0, 50.0, 2000.0, 2000.0],  # upper
        )

        popt, _ = curve_fit(_model_reparam, x, y, p0=p0, bounds=bounds, maxfev=30000)

        y_min_hat, delta_hat, r_opt_hat, width_hat = popt
        ymax_hat = float(y_min_hat) + float(delta_hat)


        results.append(
            FitResult(
                region=int(r),
                ymax=float(ymax_hat),
                r_opt_mm=float(r_opt_hat),
                width_mm=float(width_hat),
                y_min=float(y_min_hat),
            )
        )

    return results
