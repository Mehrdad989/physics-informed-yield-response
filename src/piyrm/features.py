from __future__ import annotations

import numpy as np


def generate_region_features(
    *,
    seed: int,
    n_regions: int,
) -> dict[str, np.ndarray]:
    """
    Synthetic region-level covariates (placeholders for real data later).

    Returns a dict-of-arrays with length n_regions:
      - region: int id
      - aridity_index: [0, 1] (higher = wetter / less arid)
      - soil_whc: [50, 250] (water holding capacity proxy, mm)
      - heat_stress: [0, 1] (higher = hotter / more stress)
    """
    rng = np.random.default_rng(seed)

    region = np.arange(n_regions, dtype=int)
    aridity_index = rng.uniform(0.0, 1.0, size=n_regions)
    soil_whc = rng.uniform(50.0, 250.0, size=n_regions)
    heat_stress = rng.uniform(0.0, 1.0, size=n_regions)

    return {
        "region": region,
        "aridity_index": aridity_index,
        "soil_whc": soil_whc,
        "heat_stress": heat_stress,
    }
