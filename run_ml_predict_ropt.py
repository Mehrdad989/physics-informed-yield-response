import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from piyrm.features import generate_region_features
from piyrm.fit import fit_hump_by_region
from piyrm.synthetic import generate_synthetic_region_yield


def build_feature_matrix(feat: dict[str, np.ndarray]) -> np.ndarray:
    # X shape: (n_regions, n_features)
    return np.column_stack(
        [
            feat["aridity_index"],
            feat["soil_whc"],
            feat["heat_stress"],
        ]
    )


# 1) Generate region features
n_regions = 60
feat = generate_region_features(seed=0, n_regions=n_regions)

# 2) Generate synthetic yield observations using feature-driven parameters
d = generate_synthetic_region_yield(
    seed=1,
    n_regions=n_regions,
    n_per_region=400,
    sigma_y=0.6,
    region_features=feat,
    include_true_params=True,
)

# 3) Fit curve per region to get estimated parameters (what we would do with real data)
fits = fit_hump_by_region(region=d["region"], rain_mm=d["rain_mm"], yield_obs=d["yield_obs"])

# 4) Build training targets: fitted r_opt per region
r_opt_hat = np.array([fr.r_opt_mm for fr in sorted(fits, key=lambda x: x.region)], dtype=float)

# Optional: compare to truth (useful sanity check)
r_opt_true = np.array([float(d["r_opt_true"][d["region"] == r][0]) for r in range(n_regions)], dtype=float)

# 5) Train/test split on regions
rng = np.random.default_rng(42)
idx = np.arange(n_regions)
rng.shuffle(idx)
split = int(0.8 * n_regions)
train_idx, test_idx = idx[:split], idx[split:]

X = build_feature_matrix(feat)

model = LinearRegression()
model.fit(X[train_idx], r_opt_hat[train_idx])

pred = model.predict(X[test_idx])

print("Predict r_opt from features (LinearRegression)")
print(f"Test R^2: {r2_score(r_opt_hat[test_idx], pred):.3f}")
print(f"Test MAE (mm): {mean_absolute_error(r_opt_hat[test_idx], pred):.1f}")

print("\nSanity: fitted vs true r_opt (overall)")
print(f"MAE(fit vs true) (mm): {mean_absolute_error(r_opt_true, r_opt_hat):.1f}")
