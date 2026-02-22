from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm
import arviz as az


@dataclass(frozen=True)
class BayesSaturatingMLPriorsResult:
    idata: az.InferenceData
    ymax_post_mean: np.ndarray  # shape (n_regions,)
    k_post_mean: np.ndarray     # shape (n_regions,)
    sigma_y_post_mean: float
    sigma_log_delta_post_mean: float
    sigma_logk_post_mean: float


def fit_saturating_bayes_with_ml_priors(
    *,
    region: np.ndarray,
    rain_mm: np.ndarray,
    yield_obs: np.ndarray,
    ymax_ml: np.ndarray,
    k_ml: np.ndarray,
    y_min_fixed: float = 0.5,
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    cores: int = 1,
    target_accept: float = 0.9,
    random_seed: int = 0,
    progressbar: bool = True,
) -> BayesSaturatingMLPriorsResult:
    """
    Bayesian inference for saturating curve with ML-centered priors.

    Model:
      delta_r = ymax_r - y_min  (delta_r > 0)
      log(delta_r) ~ Normal(log(delta_ml_r), sigma_log_delta)
      log(k_r)     ~ Normal(log(k_ml_r), sigma_logk)

      y_i ~ Normal( y_min + (ymax_region[i]-y_min)*(1-exp(-k_region[i]*rain_i)), sigma_y )

    Returns full InferenceData + posterior means for convenience.
    """
    region = np.asarray(region, dtype=int)
    rain_mm = np.asarray(rain_mm, dtype=float)
    yield_obs = np.asarray(yield_obs, dtype=float)

    ymax_ml = np.asarray(ymax_ml, dtype=float)
    k_ml = np.asarray(k_ml, dtype=float)

    if not (region.shape == rain_mm.shape == yield_obs.shape):
        raise ValueError("region, rain_mm, yield_obs must have the same shape")
    if region.ndim != 1:
        raise ValueError("inputs must be 1D arrays")

    n_regions = int(region.max()) + 1
    if ymax_ml.shape != (n_regions,) or k_ml.shape != (n_regions,):
        raise ValueError("ymax_ml and k_ml must have shape (n_regions,) where n_regions = max(region)+1")

    if y_min_fixed < 0:
        raise ValueError("y_min_fixed must be >= 0")

    # Safe prior centers
    k_ml = np.clip(k_ml, 1e-6, 10.0)
    delta_ml = np.clip(ymax_ml - y_min_fixed, 1e-6, None)

    log_delta_mu = np.log(delta_ml)
    logk_mu = np.log(k_ml)

    with pm.Model() as model:
        sigma_log_delta = pm.HalfNormal("sigma_log_delta", sigma=0.6)
        sigma_logk = pm.HalfNormal("sigma_logk", sigma=0.6)

        log_delta = pm.Normal("log_delta", mu=log_delta_mu, sigma=sigma_log_delta, shape=n_regions)
        logk = pm.Normal("logk", mu=logk_mu, sigma=sigma_logk, shape=n_regions)

        delta = pm.Deterministic("delta", pm.math.exp(log_delta))
        k = pm.Deterministic("k", pm.math.exp(logk))
        ymax = pm.Deterministic("ymax", y_min_fixed + delta)

        sigma_y = pm.HalfNormal("sigma_y", sigma=2.0)

        mu_y = y_min_fixed + (ymax[region] - y_min_fixed) * (1.0 - pm.math.exp(-k[region] * rain_mm))
        pm.Normal("y_obs", mu=mu_y, sigma=sigma_y, observed=yield_obs)

        idata = pm.sample(
            draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    ymax_post = idata.posterior["ymax"].mean(("chain", "draw")).values
    k_post = idata.posterior["k"].mean(("chain", "draw")).values

    sigma_y_mean = float(idata.posterior["sigma_y"].mean(("chain", "draw")).values)
    sigma_log_delta_mean = float(idata.posterior["sigma_log_delta"].mean(("chain", "draw")).values)
    sigma_logk_mean = float(idata.posterior["sigma_logk"].mean(("chain", "draw")).values)

    return BayesSaturatingMLPriorsResult(
        idata=idata,
        ymax_post_mean=ymax_post,
        k_post_mean=k_post,
        sigma_y_post_mean=sigma_y_mean,
        sigma_log_delta_post_mean=sigma_log_delta_mean,
        sigma_logk_post_mean=sigma_logk_mean,
    )