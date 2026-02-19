import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from piyrm.features import generate_region_features
from piyrm.fit import fit_hump_by_region
from piyrm.synthetic import generate_synthetic_region_yield


def X_from_features(feat: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([feat["aridity_index"], feat["soil_whc"], feat["heat_stress"]])


n_regions = 80
feat = generate_region_features(seed=10, n_regions=n_regions)

d = generate_synthetic_region_yield(
    seed=11,
    n_regions=n_regions,
    n_per_region=400,
    sigma_y=0.7,
    region_features=feat,
    include_true_params=True,
)

fits = sorted(
    fit_hump_by_region(region=d["region"], rain_mm=d["rain_mm"], yield_obs=d["yield_obs"]),
    key=lambda fr: fr.region,
)

# Targets from fitted parameters (what we'd have from real data)
Y_hat = np.column_stack([[fr.ymax for fr in fits], [fr.r_opt_mm for fr in fits], [fr.width_mm for fr in fits]])
param_names = ["ymax", "r_opt_mm", "width_mm"]

# True parameters (sanity check)
Y_true = np.column_stack(
    [
        [float(d["ymax_true"][d["region"] == r][0]) for r in range(n_regions)],
        [float(d["r_opt_true"][d["region"] == r][0]) for r in range(n_regions)],
        [float(d["width_true"][d["region"] == r][0]) for r in range(n_regions)],
    ]
)

X = X_from_features(feat)

# Train/test split on regions
rng = np.random.default_rng(0)
idx = np.arange(n_regions)
rng.shuffle(idx)
split = int(0.8 * n_regions)
train_idx, test_idx = idx[:split], idx[split:]

model = LinearRegression()
model.fit(X[train_idx], Y_hat[train_idx])

pred = model.predict(X[test_idx])

print("Multi-target parameter prediction (LinearRegression)")
for j, name in enumerate(param_names):
    r2 = r2_score(Y_hat[test_idx, j], pred[:, j])
    mae = mean_absolute_error(Y_hat[test_idx, j], pred[:, j])
    print(f"{name:9s} | Test R^2: {r2:6.3f} | Test MAE: {mae:8.3f}")

print("\nSanity: fitted vs true (overall MAE)")
for j, name in enumerate(param_names):
    mae_ft = mean_absolute_error(Y_true[:, j], Y_hat[:, j])
    print(f"{name:9s} | MAE(fit vs true): {mae_ft:8.3f}")
