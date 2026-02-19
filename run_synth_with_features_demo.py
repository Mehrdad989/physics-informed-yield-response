import pandas as pd

from piyrm.features import generate_region_features
from piyrm.synthetic import generate_synthetic_region_yield

feat = generate_region_features(seed=0, n_regions=5)

d = generate_synthetic_region_yield(
    seed=0,
    n_regions=5,
    n_per_region=3,
    region_features=feat,
    include_true_params=True,
)

df = pd.DataFrame(d)
print(df[["region", "rain_mm", "yield_obs", "ymax_true", "r_opt_true", "width_true"]].head(10))
