import numpy as np
import pymc as pm
import arviz as az
from sklearn.metrics import mean_absolute_error

from piyrm.features import generate_region_features
from piyrm.ml import build_region_feature_matrix, fit_linear_multioutput
from piyrm.fit import fit_saturating_by_region
from piyrm.synthetic import generate_synthetic_region_yield


def main():
    # ----- Setup -----
    n_regions = 40
    y_min_fixed = 0.5  # keep fixed for v1 integration (we can infer later)

    feat = generate_region_features(seed=100, n_regions=n_regions)

    # Create an intentionally "data-sparse" subset of regions
    sparse_regions = set(range(30, 40))  # last 10 regions are sparse
    n_dense = 200
    n_sparse = 35

    # Build a dataset with variable n_per_region
    all_rows = []
    for r in range(n_regions):
        npr = n_sparse if r in sparse_regions else n_dense
        d_r = generate_synthetic_region_yield(
            seed=200 + r,
            n_regions=1,              # generate one-region chunk
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
        # region id becomes 0 in each chunk; override to real region id
        d_r["region"] = np.full_like(d_r["region"], r)
        all_rows.append(d_r)

    # Concatenate dict-of-arrays
    d = {k: np.concatenate([chunk[k] for chunk in all_rows]) for k in all_rows[0].keys()}

    region = d["region"].astype(int)
    rain = d["rain_mm"].astype(float)
    y_obs = d["yield_obs"].astype(float)

    # True region parameters for evaluation
    ymax_true = np.array([float(d["ymax_true"][region == r][0]) for r in range(n_regions)], dtype=float)
    k_true = np.array([float(d["k_true"][region == r][0]) for r in range(n_regions)], dtype=float)

    # ----- Step A: Get per-region fitted params (like "classical" Stage 1) -----
    fits = sorted(
        fit_saturating_by_region(region=region, rain_mm=rain, yield_obs=y_obs),
        key=lambda fr: fr.region,
    )
    ymax_hat = np.array([fr.ymax for fr in fits], dtype=float)
    k_hat = np.array([fr.k for fr in fits], dtype=float)

    # ----- Step B: Train ML on dense regions only -----
    X = build_region_feature_matrix(feat)

    dense_idx = np.array([r for r in range(n_regions) if r not in sparse_regions], dtype=int)
    # ML targets are fitted params from dense regions
    Y_dense = np.column_stack([ymax_hat[dense_idx], k_hat[dense_idx]])

    ml_model = fit_linear_multioutput(X[dense_idx], Y_dense)

    # ML prior centers for ALL regions
    ymax_ml, k_ml = ml_model.predict(X).T

    # enforce positivity of k prior centers
    k_ml = np.clip(k_ml, 1e-4, 1.0)

    # ----- Step C: Bayesian model with ML-informed priors -----
    # Reparameterize ymax = y_min + delta (delta > 0), and model log(k)
    delta_ml = np.clip(ymax_ml - y_min_fixed, 1e-3, None)
    log_delta_mu = np.log(delta_ml)
    logk_mu = np.log(k_ml)

    with pm.Model() as model:
        # How strongly we trust ML (learned from data)
        sigma_log_delta = pm.HalfNormal("sigma_log_delta", sigma=0.6)
        sigma_logk = pm.HalfNormal("sigma_logk", sigma=0.6)

        # Region-specific parameters with ML-centered priors
        log_delta = pm.Normal("log_delta", mu=log_delta_mu, sigma=sigma_log_delta, shape=n_regions)
        logk = pm.Normal("logk", mu=logk_mu, sigma=sigma_logk, shape=n_regions)

        delta = pm.Deterministic("delta", pm.math.exp(log_delta))
        k = pm.Deterministic("k", pm.math.exp(logk))
        ymax = pm.Deterministic("ymax", y_min_fixed + delta)

        sigma_y = pm.HalfNormal("sigma_y", sigma=2.0)

        mu_y = y_min_fixed + (ymax[region] - y_min_fixed) * (1.0 - pm.math.exp(-k[region] * rain))
        pm.Normal("y_obs", mu=mu_y, sigma=sigma_y, observed=y_obs)

        idata = pm.sample(
            300,
            tune=300,
            chains=2,
            cores=1,
            target_accept=0.9,
            progressbar=True,
        )

    # Posterior means
    ymax_post = idata.posterior["ymax"].mean(("chain", "draw")).values
    k_post = idata.posterior["k"].mean(("chain", "draw")).values

    # ----- Evaluation: focus on sparse regions -----
    sparse_idx = np.array(sorted(sparse_regions), dtype=int)

    mae_fit_ymax = mean_absolute_error(ymax_true[sparse_idx], ymax_hat[sparse_idx])
    mae_fit_k = mean_absolute_error(k_true[sparse_idx], k_hat[sparse_idx])

    mae_post_ymax = mean_absolute_error(ymax_true[sparse_idx], ymax_post[sparse_idx])
    mae_post_k = mean_absolute_error(k_true[sparse_idx], k_post[sparse_idx])

    mae_ml_ymax = mean_absolute_error(ymax_true[sparse_idx], ymax_ml[sparse_idx])
    mae_ml_k = mean_absolute_error(k_true[sparse_idx], k_ml[sparse_idx])

    print("\n=== Sparse-region evaluation (regions 30-39) ===")
    print(f"ML prior centers:      MAE(ymax)={mae_ml_ymax:.3f}   MAE(k)={mae_ml_k:.6f}")
    print(f"Per-region fit only:   MAE(ymax)={mae_fit_ymax:.3f}   MAE(k)={mae_fit_k:.6f}")
    print(f"Bayes w/ ML priors:    MAE(ymax)={mae_post_ymax:.3f}   MAE(k)={mae_post_k:.6f}")

    print("\nBayes hyperparameters:")
    print(az.summary(idata, var_names=["sigma_log_delta", "sigma_logk", "sigma_y"]))


if __name__ == "__main__":
    main()