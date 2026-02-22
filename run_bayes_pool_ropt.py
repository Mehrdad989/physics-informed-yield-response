import numpy as np
import arviz as az
import pymc as pm

from piyrm.features import generate_region_features
from piyrm.fit import fit_hump_by_region
from piyrm.synthetic import generate_synthetic_region_yield


def main():
    # 1) Generate synthetic dataset
    n_regions = 40
    feat = generate_region_features(seed=1, n_regions=n_regions)

    d = generate_synthetic_region_yield(
        seed=2,
        n_regions=n_regions,
        n_per_region=60,
        sigma_y=1.2,
        region_features=feat,
        include_true_params=True,
    )

    # 2) Fit per region to get noisy estimates r_opt_hat
    fits = sorted(
        fit_hump_by_region(region=d["region"], rain_mm=d["rain_mm"], yield_obs=d["yield_obs"]),
        key=lambda fr: fr.region,
    )
    r_opt_hat = np.array([fr.r_opt_mm for fr in fits], dtype=float)

    # True values (for sanity check)
    r_opt_true = np.array([float(d["r_opt_true"][d["region"] == r][0]) for r in range(n_regions)], dtype=float)

    # 3) Hierarchical pooling model
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=450.0, sigma=200.0)
        tau = pm.HalfNormal("tau", sigma=100.0)

        r_opt_region = pm.Normal("r_opt_region", mu=mu, sigma=tau, shape=n_regions)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=30.0)
        pm.Normal("r_opt_hat", mu=r_opt_region, sigma=sigma_obs, observed=r_opt_hat)

        idata = pm.sample(
            700,
            tune=700,
            chains=2,
            target_accept=0.9,
            progressbar=True,
        )

    # 4) Summaries
    summ = az.summary(idata, var_names=["mu", "tau", "sigma_obs"])
    print(summ)

    # 5) Shrinkage sanity check
    post_mean_region = idata.posterior["r_opt_region"].mean(("chain", "draw")).values
    mae_hat_vs_true = np.mean(np.abs(r_opt_hat - r_opt_true))
    mae_post_vs_true = np.mean(np.abs(post_mean_region - r_opt_true))

    print(f"\nMAE raw fit r_opt_hat vs true:     {mae_hat_vs_true:.2f} mm")
    print(f"MAE pooled posterior mean vs true: {mae_post_vs_true:.2f} mm")


if __name__ == "__main__":
    main()
