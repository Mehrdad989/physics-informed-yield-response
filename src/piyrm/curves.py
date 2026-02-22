from __future__ import annotations

import numpy as np


def yield_rain_hump(
    rain_mm: np.ndarray,
    ymax: float,
    r_opt_mm: float,
    width_mm: float,
    y_min: float = 0.0,
) -> np.ndarray:
    """
    Smooth hump-shaped yield response to seasonal total rainfall.

    Equation:
        y(r) = y_min + (ymax - y_min) * exp(-0.5 * ((r - r_opt)/width)^2)

    Parameters
    ----------
    rain_mm : array-like
        Seasonal total rainfall in mm (must be >= 0).
    ymax : float
        Maximum attainable yield (must be >= y_min).
    r_opt_mm : float
        Optimal rainfall (mm) where yield is maximized.
    width_mm : float
        Sensitivity / spread parameter in mm (must be > 0).
    y_min : float, default 0.0
        Yield floor (must be >= 0).

    Returns
    -------
    np.ndarray
        Expected yield values (same shape as rain_mm), bounded in [y_min, ymax].
    """
    rain = np.asarray(rain_mm, dtype=float)

    if np.any(rain < 0):
        raise ValueError("rain_mm must be >= 0")
    if width_mm <= 0:
        raise ValueError("width_mm must be > 0")
    if y_min < 0:
        raise ValueError("y_min must be >= 0")
    if ymax < y_min:
        raise ValueError("ymax must be >= y_min")

    z = (rain - float(r_opt_mm)) / float(width_mm)
    y = float(y_min) + (float(ymax) - float(y_min)) * np.exp(-0.5 * z * z)
    return y

def yield_rain_saturating(
    rain_mm: np.ndarray,
    ymax: float,
    k: float,
    y_min: float = 0.0,
) -> np.ndarray:
    """
    Monotonic saturating yield response to seasonal total rainfall.

    Equation:
        y(r) = y_min + (ymax - y_min) * (1 - exp(-k * r))

    Parameters
    ----------
    rain_mm : array-like
        Seasonal total rainfall in mm (must be >= 0).
    ymax : float
        Maximum attainable yield (must be >= y_min).
    k : float
        Responsiveness parameter (must be > 0). Larger k = saturates faster.
    y_min : float, default 0.0
        Yield floor (must be >= 0).

    Returns
    -------
    np.ndarray
        Expected yield values (same shape as rain_mm), bounded in [y_min, ymax].
    """
    rain = np.asarray(rain_mm, dtype=float)

    if np.any(rain < 0):
        raise ValueError("rain_mm must be >= 0")
    if k <= 0:
        raise ValueError("k must be > 0")
    if y_min < 0:
        raise ValueError("y_min must be >= 0")
    if ymax < y_min:
        raise ValueError("ymax must be >= y_min")

    y = float(y_min) + (float(ymax) - float(y_min)) * (1.0 - np.exp(-float(k) * rain))
    return y