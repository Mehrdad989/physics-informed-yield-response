import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

from piyrm.features import generate_region_features
from piyrm.ml import build_region_feature_matrix, fit_linear_multioutput
from piyrm.fit import fit_saturating_by_region
from piyrm.synthetic import generate_synthetic_region_yield


n_regions = 80
feat = generate_region_features(seed=21, n_regions=n_regions)

d = generate_synthetic_region_yield(
    seed=22,
    n_regions=n_regions,
    n_per_region=400,
    sigma_y=0.7,
    region_features=feat,
    include_true_params=True,
    curve_family="saturating",
)

fits = sorted(
    fit_saturating_by_region(region=d["region"], rain_mm=d["rain_mm"], yield_obs=d["yield_obs"]),
    key=lambda fr: fr.region,
)

# targets from fitted params
ymax_hat = np.array([fr.ymax for fr in fits], dtype=float)
k_hat = np.array([fr.k for fr in fits], dtype=float)

Y_hat = np.column_stack([ymax_hat, k_hat])
names = ["ymax", "k"]

X = build_region_feature_matrix(feat)

# deterministic split
train_idx = np.arange(0, 60)
test_idx = np.arange(60, n_regions)

model = fit_linear_multioutput(X[train_idx], Y_hat[train_idx])
pred = model.predict(X[test_idx])

print("Saturating parameter prediction (LinearRegression)")
for j, name in enumerate(names):
    r2 = r2_score(Y_hat[test_idx, j], pred[:, j])
    mae = mean_absolute_error(Y_hat[test_idx, j], pred[:, j])
    print(f"{name:5s} | Test R^2: {r2:6.3f} | Test MAE: {mae:.6f}")