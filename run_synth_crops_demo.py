import pandas as pd
from piyrm.synthetic_crops import generate_synthetic_region_crop_yield

d = generate_synthetic_region_crop_yield(seed=0, n_regions=3, n_per_region_crop=5)
df = pd.DataFrame(d)

print(df[["region", "crop", "rain_mm", "yield_obs", "ymax_true", "k_true"]].head(12))
print("\nRows:", len(df), "| unique regions:", df["region"].nunique(), "| unique crops:", df["crop"].nunique())