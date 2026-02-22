import numpy as np
import pandas as pd

from piyrm.fit import fit_saturating_by_region
from piyrm.synthetic import generate_synthetic_region_yield

d = generate_synthetic_region_yield(
    seed=0,
    n_regions=3,
    n_per_region=500,
    sigma_y=0.6,
    include_true_params=True,
    curve_family="saturating",
)

df = pd.DataFrame(d)
fits = fit_saturating_by_region(region=df["region"], rain_mm=df["rain_mm"], yield_obs=df["yield_obs"])

for fr in fits:
    true_k = float(df[df["region"] == fr.region]["k_true"].iloc[0])
    true_ymax = float(df[df["region"] == fr.region]["ymax_true"].iloc[0])
    true_ymin = float(df[df["region"] == fr.region]["y_min_true"].iloc[0])

    print(f"Region {fr.region}")
    print(f"  TRUE  ymax={true_ymax:.3f}  k={true_k:.5f}  y_min={true_ymin:.3f}")
    print(f"  FIT   ymax={fr.ymax:.3f}  k={fr.k:.5f}  y_min={fr.y_min:.3f}")
    print()