from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt


@dataclass(frozen=True)
class BayesCrossCropMLPriorsResult:
    idata: az.InferenceData
    ymax_post_mean: np.ndarray  # shape (n_crops, n_regions)
    k_post_mean: np.ndarray     # shape (n_crops, n_regions)


def fit_crosscrop_saturating_bayes_with_ml_priors(
    *,
    crop_id: np.ndarray,
    region: np.ndarray,
    rain_mm: np.ndarray,
    yield_obs: np.ndarray,
    ymax_ml: np.ndarray,  # shape (n_crops, n_regions)
    k_ml: np.ndarray,     # shape (n_crops, n_regions)
    y_min_fixed: float = 0.5,
    draws: int = 250,
    tune: int = 250,
    chains: int = 2,
    cores: int = 1,
    target_accept: float = 0.9,
    random_seed: int = 123,
    progressbar: bool = True,
) -> BayesCrossCropMLPriorsResult:
    """
    Cross-crop Bayesian model for saturating curve with ML-centered priors.

    Parameters are indexed by (crop, region). Priors:
      log(delta_{c,r}) ~ Normal(log(delta_ml_{c,r}) + crop_shift_delta[c], sigma_log_delta_global)
      log(k_{c,r})     ~ Normal(log(k_ml_{c,r})     + crop_shift_k[c],     sigma_logk_global)

    Returns full idata + posterior means for (ymax, k).
    """
    crop_id = np.asarray(crop_id, dtype=int)
    region = np.asarray(region, dtype=int)
    rain_mm = np.asarray(rain_mm, dtype=float)
    yield_obs = np.asarray(yield_obs, dtype=float)

    if not (crop_id.shape == region.shape == rain_mm.shape == yield_obs.shape):
        raise ValueError("crop_id, region, rain_mm, yield_obs must have the same shape")
    if crop_id.ndim != 1:
        raise ValueError("inputs must be 1D arrays")

    n_crops = int(crop_id.max()) + 1
    n_regions = int(region.max()) + 1

    ymax_ml = np.asarray(ymax_ml, dtype=float)
    k_ml = np.asarray(k_ml, dtype=float)
    if ymax_ml.shape != (n_crops, n_regions) or k_ml.shape != (n_crops, n_regions):
        raise ValueError("ymax_ml and k_ml must have shape (n_crops, n_regions)")

    if y_min_fixed < 0:
        raise ValueError("y_min_fixed must be >= 0")

    k_ml = np.clip(k_ml, 1e-6, 10.0)
    delta_ml = np.clip(ymax_ml - y_min_fixed, 1e-6, None)

    log_delta_mu = np.log(delta_ml).reshape(-1)  # (n_crops*n_regions,)
    logk_mu = np.log(k_ml).reshape(-1)

    # Flatten crop-region index
    cr_index = crop_id * n_regions + region
    n_cr = n_crops * n_regions

    with pm.Model() as model:
        sigma_log_delta_global = pm.HalfNormal("sigma_log_delta_global", sigma=0.6)
        sigma_logk_global = pm.HalfNormal("sigma_logk_global", sigma=0.6)

        crop_delta_shift = pm.Normal("crop_delta_shift", mu=0.0, sigma=0.2, shape=n_crops)
        crop_logk_shift = pm.Normal("crop_logk_shift", mu=0.0, sigma=0.2, shape=n_crops)

        crop_shift_delta_flat = pt.repeat(crop_delta_shift, n_regions)
        crop_shift_logk_flat = pt.repeat(crop_logk_shift, n_regions)

        log_delta = pm.Normal(
            "log_delta",
            mu=log_delta_mu + crop_shift_delta_flat,
            sigma=sigma_log_delta_global,
            shape=n_cr,
        )
        logk = pm.Normal(
            "logk",
            mu=logk_mu + crop_shift_logk_flat,
            sigma=sigma_logk_global,
            shape=n_cr,
        )

        delta = pm.Deterministic("delta", pm.math.exp(log_delta))
        k = pm.Deterministic("k", pm.math.exp(logk))
        ymax = pm.Deterministic("ymax", y_min_fixed + delta)

        sigma_y = pm.HalfNormal("sigma_y", sigma=2.0)

        mu_y = y_min_fixed + (ymax[cr_index] - y_min_fixed) * (1.0 - pm.math.exp(-k[cr_index] * rain_mm))
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

    ymax_post = idata.posterior["ymax"].mean(("chain", "draw")).values.reshape(n_crops, n_regions)
    k_post = idata.posterior["k"].mean(("chain", "draw")).values.reshape(n_crops, n_regions)

    return BayesCrossCropMLPriorsResult(idata=idata, ymax_post_mean=ymax_post, k_post_mean=k_post)