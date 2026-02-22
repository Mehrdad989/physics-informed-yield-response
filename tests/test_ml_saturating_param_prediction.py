import numpy as np
from sklearn.metrics import r2_score

from piyrm.features import generate_region_features
from piyrm.fit import fit_saturating_by_region
from piyrm.ml import build_region_feature_matrix, fit_linear_multioutput
from piyrm.synthetic import generate_synthetic_region_yield


def test_linear_model_predicts_saturating_params_on_synthetic():
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

    ymax_hat = np.array([fr.ymax for fr in fits], dtype=float)
    k_hat = np.array([fr.k for fr in fits], dtype=float)
    Y_hat = np.column_stack([ymax_hat, k_hat])

    X = build_region_feature_matrix(feat)

    # deterministic split for stability
    train_idx = np.arange(0, 60)
    test_idx = np.arange(60, n_regions)

    model = fit_linear_multioutput(X[train_idx], Y_hat[train_idx])
    pred = model.predict(X[test_idx])

    r2_ymax = r2_score(Y_hat[test_idx, 0], pred[:, 0])
    r2_k = r2_score(Y_hat[test_idx, 1], pred[:, 1])

    # thresholds chosen to be stable for this synthetic setup
    assert r2_ymax > 0.93
    assert r2_k > 0.90