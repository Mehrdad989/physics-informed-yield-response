import numpy as np
from sklearn.metrics import mean_absolute_error

from piyrm.features import generate_region_features
from piyrm.synthetic_crops import generate_synthetic_region_crop_yield
from piyrm.fit import fit_saturating_by_crop_region
from piyrm.ml import fit_linear_multioutput
from piyrm.bayes_crops import fit_crosscrop_saturating_bayes_with_ml_priors


def main():
    # Settings
    n_regions = 30
    n_crops = 2
    y_min_fixed = 0.5

    sparse_regions = set(range(24, 30))
    dense_n = 120
    sparse_n = 30

    # 1) Region features
    feat = generate_region_features(seed=0, n_regions=n_regions)

    # 2) Build variable-n dataset by concatenating per-region chunks
    chunks = []
    for r in range(n_regions):
        npr = sparse_n if r in sparse_regions else dense_n
        d_r = generate_synthetic_region_crop_yield(
            seed=100 + r,
            n_regions=1,
            n_per_region_crop=npr,
            sigma_y=0.8,
            region_features={
                "region": np.array([0], dtype=int),
                "aridity_index": feat["aridity_index"][r:r+1],
                "soil_whc": feat["soil_whc"][r:r+1],
                "heat_stress": feat["heat_stress"][r:r+1],
            },
        )
        d_r["region"] = np.full_like(d_r["region"], r)
        chunks.append(d_r)

    d = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0].keys()}

    crop_id = d["crop_id"].astype(int)
    region = d["region"].astype(int)
    rain = d["rain_mm"].astype(float)
    y_obs = d["yield_obs"].astype(float)

    # True params for evaluation
    ymax_true = np.zeros((n_crops, n_regions), dtype=float)
    k_true = np.zeros((n_crops, n_regions), dtype=float)
    for c in range(n_crops):
        for r in range(n_regions):
            m = (crop_id == c) & (region == r)
            ymax_true[c, r] = float(d["ymax_true"][m][0])
            k_true[c, r] = float(d["k_true"][m][0])

    # 3) Stage 1: per (crop, region) fits
    fits = fit_saturating_by_crop_region(crop_id=crop_id, region=region, rain_mm=rain, yield_obs=y_obs)
    ymax_hat = np.full((n_crops, n_regions), np.nan, dtype=float)
    k_hat = np.full((n_crops, n_regions), np.nan, dtype=float)
    for fr in fits:
        ymax_hat[fr.crop_id, fr.region] = fr.ymax
        k_hat[fr.crop_id, fr.region] = fr.k

    # 4) Stage 2: ML priors trained on dense regions only
    crop_rows = np.repeat(np.arange(n_crops, dtype=int), n_regions)
    region_rows = np.tile(np.arange(n_regions, dtype=int), n_crops)

    Y_rows = np.column_stack([ymax_hat.reshape(-1), k_hat.reshape(-1)])
    X_rows = np.column_stack(
        [
            feat["aridity_index"][region_rows],
            feat["soil_whc"][region_rows],
            feat["heat_stress"][region_rows],
            crop_rows,
        ]
    )

    dense_mask = np.array([r not in sparse_regions for r in region_rows], dtype=bool)
    ml = fit_linear_multioutput(X_rows[dense_mask], Y_rows[dense_mask])

    Y_ml = ml.predict(X_rows)
    ymax_ml = Y_ml[:, 0].reshape(n_crops, n_regions)
    k_ml = np.clip(Y_ml[:, 1].reshape(n_crops, n_regions), 1e-6, 10.0)

    # 5) Stage 3: Bayes with ML priors
    res = fit_crosscrop_saturating_bayes_with_ml_priors(
        crop_id=crop_id,
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
        progressbar=True,
    )

    # 6) Compare on sparse regions
    sparse_idx = np.array(sorted(sparse_regions), dtype=int)

    mae_fit_ymax = 0.5 * (
        mean_absolute_error(ymax_true[0, sparse_idx], ymax_hat[0, sparse_idx])
        + mean_absolute_error(ymax_true[1, sparse_idx], ymax_hat[1, sparse_idx])
    )
    mae_bayes_ymax = 0.5 * (
        mean_absolute_error(ymax_true[0, sparse_idx], res.ymax_post_mean[0, sparse_idx])
        + mean_absolute_error(ymax_true[1, sparse_idx], res.ymax_post_mean[1, sparse_idx])
    )

    mae_fit_k = 0.5 * (
        mean_absolute_error(k_true[0, sparse_idx], k_hat[0, sparse_idx])
        + mean_absolute_error(k_true[1, sparse_idx], k_hat[1, sparse_idx])
    )
    mae_bayes_k = 0.5 * (
        mean_absolute_error(k_true[0, sparse_idx], res.k_post_mean[0, sparse_idx])
        + mean_absolute_error(k_true[1, sparse_idx], res.k_post_mean[1, sparse_idx])
    )

    print("\n=== Synthetic end-to-end pipeline ===")
    print(f"Sparse regions: {sorted(sparse_regions)}")
    print(f"Fit-only    MAE(ymax)={mae_fit_ymax:.3f}   MAE(k)={mae_fit_k:.6f}")
    print(f"Bayes+ML    MAE(ymax)={mae_bayes_ymax:.3f}   MAE(k)={mae_bayes_k:.6f}")


if __name__ == "__main__":
    main()