import pandas as pd
from piyrm.synthetic_crops import generate_synthetic_region_crop_yield
from piyrm.fit import fit_saturating_by_crop_region

d = generate_synthetic_region_crop_yield(seed=0, n_regions=3, n_per_region_crop=200, sigma_y=0.7)
df = pd.DataFrame(d)

fits = fit_saturating_by_crop_region(
    crop_id=df["crop_id"].to_numpy(),
    region=df["region"].to_numpy(),
    rain_mm=df["rain_mm"].to_numpy(),
    yield_obs=df["yield_obs"].to_numpy(),
)

# Show a few
for fr in fits:
    sub = df[(df["crop_id"] == fr.crop_id) & (df["region"] == fr.region)].iloc[0]
    print(f"crop={sub['crop']} region={fr.region}")
    print(f"  TRUE ymax={sub['ymax_true']:.3f}  k={sub['k_true']:.5f}")
    print(f"  FIT  ymax={fr.ymax:.3f}  k={fr.k:.5f}")
    print()