import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from piyrm.features import generate_region_features
from piyrm.synthetic_crops import generate_synthetic_region_crop_yield, CROP_NAMES
from piyrm.fit import fit_saturating_by_crop_region


def build_X(region_features: dict[str, np.ndarray], crop_id: np.ndarray, region: np.ndarray) -> np.ndarray:
    # join per-row region covariates + crop_id
    aridity = region_features["aridity_index"][region]
    soil = region_features["soil_whc"][region]
    heat = region_features["heat_stress"][region]
    return np.column_stack([aridity, soil, heat, crop_id])


# 1) Region features
n_regions = 60
feat = generate_region_features(seed=0, n_regions=n_regions)

# 2) Synthetic crop×region yield dataset
d = generate_synthetic_region_crop_yield(
    seed=1,
    n_regions=n_regions,
    n_per_region_crop=250,
    sigma_y=0.8,
    region_features=feat,   # <<< add this
)

df = pd.DataFrame(d)

# 3) Fit parameters per (crop, region)
fits = fit_saturating_by_crop_region(
    crop_id=df["crop_id"].to_numpy(),
    region=df["region"].to_numpy(),
    rain_mm=df["rain_mm"].to_numpy(),
    yield_obs=df["yield_obs"].to_numpy(),
)

# turn fit results into training rows
crop_ids = np.array([fr.crop_id for fr in fits], dtype=int)
regions = np.array([fr.region for fr in fits], dtype=int)

ymax_hat = np.array([fr.ymax for fr in fits], dtype=float)
k_hat = np.array([fr.k for fr in fits], dtype=float)

Y = np.column_stack([ymax_hat, k_hat])
X = build_X(feat, crop_ids, regions)

# 4) Train/test split by REGION (cross-region transfer-style)
rng = np.random.default_rng(42)
region_ids = np.arange(n_regions)
rng.shuffle(region_ids)
split = int(0.8 * n_regions)
train_regions = set(region_ids[:split])
test_regions = set(region_ids[split:])

train_mask = np.array([r in train_regions for r in regions], dtype=bool)
test_mask = ~train_mask

model = LinearRegression()
model.fit(X[train_mask], Y[train_mask])

pred = model.predict(X[test_mask])

print("Cross-crop parameter prediction (LinearRegression)")
for j, name in enumerate(["ymax", "k"]):
    r2 = r2_score(Y[test_mask, j], pred[:, j])
    mae = mean_absolute_error(Y[test_mask, j], pred[:, j])
    print(f"{name:4s} | Test R^2: {r2:6.3f} | Test MAE: {mae:.6f}")

# Optional: show a couple predictions
print("\nExample predictions (first 5 test rows):")
test_idx = np.where(test_mask)[0][:5]
for i in test_idx:
    crop = CROP_NAMES[crop_ids[i]]
    r = regions[i]
    print(
        f"region={r:02d} crop={crop:5s} | true_ymax={Y[i,0]:.2f} pred_ymax={pred[np.where(test_idx==i)[0][0],0]:.2f} | "
        f"true_k={Y[i,1]:.4f} pred_k={pred[np.where(test_idx==i)[0][0],1]:.4f}"
    )