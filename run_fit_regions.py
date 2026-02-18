import pandas as pd

from piyrm.fit import fit_hump_by_region
from piyrm.synthetic import generate_synthetic_region_yield

d = generate_synthetic_region_yield(seed=7, n_regions=3, n_per_region=400, sigma_y=0.5, include_true_params=True)
df = pd.DataFrame(d)

fits = fit_hump_by_region(
    region=df["region"].to_numpy(),
    rain_mm=df["rain_mm"].to_numpy(),
    yield_obs=df["yield_obs"].to_numpy(),
)

for fr in fits:
    sub = df[df["region"] == fr.region].iloc[0]
    print(f"Region {fr.region}")
    print(f"  TRUE  ymax={sub['ymax_true']:.3f}  r_opt={sub['r_opt_true']:.1f}  width={sub['width_true']:.1f}  y_min={sub['y_min_true']:.3f}")
    print(f"  FIT   ymax={fr.ymax:.3f}  r_opt={fr.r_opt_mm:.1f}  width={fr.width_mm:.1f}  y_min={fr.y_min:.3f}")
    print()
