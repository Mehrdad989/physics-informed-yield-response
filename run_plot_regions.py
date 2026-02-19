import pandas as pd
import matplotlib.pyplot as plt

from piyrm.synthetic import generate_synthetic_region_yield

d = generate_synthetic_region_yield(seed=7, n_regions=3, n_per_region=200, sigma_y=0.5)
df = pd.DataFrame(d)

for region_id in sorted(df["region"].unique()):
    sub = df[df["region"] == region_id]
    plt.figure()
    plt.scatter(sub["rain_mm"], sub["yield_obs"], s=10)
    plt.title(f"Region {region_id}: yield vs rainfall (synthetic)")
    plt.xlabel("Seasonal rainfall (mm)")
    plt.ylabel("Observed yield (t/ha)")

plt.show()
