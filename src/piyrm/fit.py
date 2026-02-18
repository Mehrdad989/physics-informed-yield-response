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


def _model(rain_mm: np.ndarray, ymax: float, r_opt_mm: float, width_mm: float, y_min: float) -> np.ndarray:
    return yield_rain_hump(
        rain_mm=np.asarray(rain_mm),
        ymax=float(ymax),
        r_opt_mm=float(r_opt_mm),
        width_mm=float(width_mm),
        y_min=float(y_min),
    )


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

        p0 = [
            float(np.percentile(y, 95)),  # ymax
            float(np.median(x)),          # r_opt
            float(p0_width_mm),           # width
            float(np.percentile(y, 5)),   # y_min
        ]

        popt, _ = curve_fit(_model, x, y, p0=p0, bounds=bounds, maxfev=20000)
        ymax_hat, r_opt_hat, width_hat, y_min_hat = popt

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
