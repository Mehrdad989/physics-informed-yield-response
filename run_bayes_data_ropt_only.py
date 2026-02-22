import numpy as np
import arviz as az
import pymc as pm

from piyrm.features import generate_region_features
from piyrm.synthetic import generate_synthetic_region_yield
from piyrm.curves import yield_rain_hump


def main():
    # Keep it small for first Bayes model
    n_regions = 16
    n_per_region = 120

    feat = generate_region_features(seed=3, n_regions=n_regions)
    d = generate_synthetic_region_yield(
        seed=4,
        n_regions=n_regions,
        n_per_region=n_per_region,
        sigma_y=0.8,
        region_features=feat,
        include_true_params=True,
    )

    region = d["region"].astype(int)
    rain = d["rain_mm"].astype(float)
    y_obs = d["yield_obs"].astype(float)

    # We'll keep these fixed (simple first model)
    # Use the average true values (in real life you wouldn't know these)
    ymax_fixed = float(np.mean([d["ymax_true"][region == r][0] for r in range(n_regions)]))
    width_fixed = float(np.mean([d["width_true"][region == r][0] for r in range(n_regions)]))
    y_min_fixed = float(np.mean([d["y_min_true"][region == r][0] for r in range(n_regions)]))

    r_opt_true = np.array([float(d["r_opt_true"][region == r][0]) for r in range(n_regions)], dtype=float)

    with pm.Model() as model:
        mu_ropt = pm.Normal("mu_ropt", mu=450.0, sigma=200.0)
        tau_ropt = pm.HalfNormal("tau_ropt", sigma=120.0)

        r_opt_region = pm.Normal("r_opt_region", mu=mu_ropt, sigma=tau_ropt, shape=n_regions)

        sigma_y = pm.HalfNormal("sigma_y", sigma=2.0)

        # Expected yield per observation
        mu_y = y_min_fixed + (ymax_fixed - y_min_fixed) * pm.math.exp(
            -0.5 * ((rain - r_opt_region[region]) / width_fixed) ** 2
        )

        pm.Normal("y_obs", mu=mu_y, sigma=sigma_y, observed=y_obs)

        idata = pm.sample(800, tune=800, chains=2,cores=1, target_accept=0.9, progressbar=True)

    print(az.summary(idata, var_names=["mu_ropt", "tau_ropt", "sigma_y"]))

    post_mean_ropt = idata.posterior["r_opt_region"].mean(("chain", "draw")).values
    mae_post = np.mean(np.abs(post_mean_ropt - r_opt_true))
    print(f"\nMAE pooled posterior mean r_opt vs true: {mae_post:.2f} mm")


if __name__ == "__main__":
    main()