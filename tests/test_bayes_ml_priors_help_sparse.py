import numpy as np
from sklearn.metrics import mean_absolute_error

from piyrm.features import generate_region_features
from piyrm.fit import fit_saturating_by_region
from piyrm.ml import build_region_feature_matrix, fit_linear_multioutput
from piyrm.synthetic import generate_synthetic_region_yield
from piyrm.bayes import fit_saturating_bayes_with_ml_priors


def test_bayes_with_ml_priors_improves_sparse_regions():
    n_regions = 40
    y_min_fixed = 0.5

    feat = generate_region_features(seed=100, n_regions=n_regions)

    sparse_regions = set(range(30, 40))
    n_dense = 200
    n_sparse = 35

    # Build variable-n dataset by concatenating one-region chunks
    chunks = []
    for r in range(n_regions):
        npr = n_sparse if r in sparse_regions else n_dense
        d_r = generate_synthetic_region_yield(
            seed=200 + r,
            n_regions=1,
            n_per_region=npr,
            sigma_y=0.8,
            region_features={
                "region": np.array([0], dtype=int),
                "aridity_index": feat["aridity_index"][r:r+1],
                "soil_whc": feat["soil_whc"][r:r+1],
                "heat_stress": feat["heat_stress"][r:r+1],
            },
            include_true_params=True,
            curve_family="saturating",
        )
        d_r["region"] = np.full_like(d_r["region"], r)
        chunks.append(d_r)

    d = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0].keys()}

    region = d["region"].astype(int)
    rain = d["rain_mm"].astype(float)
    y_obs = d["yield_obs"].astype(float)

    ymax_true = np.array([float(d["ymax_true"][region == r][0]) for r in range(n_regions)], dtype=float)
    k_true = np.array([float(d["k_true"][region == r][0]) for r in range(n_regions)], dtype=float)

    # per-region fits
    fits = sorted(
        fit_saturating_by_region(region=region, rain_mm=rain, yield_obs=y_obs),
        key=lambda fr: fr.region,
    )
    ymax_hat = np.array([fr.ymax for fr in fits], dtype=float)
    k_hat = np.array([fr.k for fr in fits], dtype=float)

    # ML trained on dense regions only
    X = build_region_feature_matrix(feat)
    dense_idx = np.array([r for r in range(n_regions) if r not in sparse_regions], dtype=int)
    Y_dense = np.column_stack([ymax_hat[dense_idx], k_hat[dense_idx]])

    ml_model = fit_linear_multioutput(X[dense_idx], Y_dense)
    ymax_ml, k_ml = ml_model.predict(X).T

    # Bayes with ML priors (keep small for test runtime)
    res = fit_saturating_bayes_with_ml_priors(
        region=region,
        rain_mm=rain,
        yield_obs=y_obs,
        ymax_ml=ymax_ml,
        k_ml=k_ml,
        y_min_fixed=y_min_fixed,
        draws=150,
        tune=150,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=123,
        progressbar=False,
    )

    sparse_idx = np.array(sorted(sparse_regions), dtype=int)

    mae_fit_ymax = mean_absolute_error(ymax_true[sparse_idx], ymax_hat[sparse_idx])
    mae_fit_k = mean_absolute_error(k_true[sparse_idx], k_hat[sparse_idx])

    mae_post_ymax = mean_absolute_error(ymax_true[sparse_idx], res.ymax_post_mean[sparse_idx])
    mae_post_k = mean_absolute_error(k_true[sparse_idx], res.k_post_mean[sparse_idx])

    # Bayes should beat per-region fit on sparse regions (robust margins)
    assert mae_post_ymax < mae_fit_ymax
    assert mae_post_k < mae_fit_k