import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def sat_yield(r_mm, ymax, k):
    return ymax * (1 - np.exp(-k * r_mm))


def load_ml_centers():
    """
    We try a few likely filenames so you don’t have to remember exactly.
    Returns dataframe with columns: state_fips, crop, log_ymax, log_k
    """
    candidates = [
        "data/processed/ml_state_joint_v2.csv",
        "data/processed/ml_state_joint.csv",
        "data/processed/ml_state_joint_loso.csv",
    ]
    for path in candidates:
        try:
            df = pd.read_csv(path, dtype={"state_fips": str, "crop": str})
            need = {"state_fips", "crop", "log_ymax", "log_k"}
            if need.issubset(df.columns):
                return df[list(need)].copy(), path
        except FileNotFoundError:
            pass
    raise FileNotFoundError(
        "Could not find ML centers file. Expected one of: "
        + ", ".join(candidates)
    )


def build_state_year_table(panel_path="data/processed/panel_county_year.csv"):
    panel = pd.read_csv(
        panel_path,
        dtype={"state_fips": str, "county_fips": str, "crop": str},
    )
    panel = panel[(panel["yield_value"] > 0) & (panel["precip_mm"].notna())].copy()

    # state-year-crop mean
    sy = (
        panel.groupby(["state_fips", "crop", "year"], as_index=False)
        .agg(y=("yield_value", "mean"), r=("precip_mm", "mean"))
    )
    return sy


def fit_global_bayes_sigmas(df_sy_ml):
    """
    Fit Bayes model ON ALL state-year observations (fast-ish),
    to get posterior for sigma_logymax, sigma_logk, sigma_y.
    We only need these sigmas to generate predictive intervals in LOSO scoring.
    """
    df = df_sy_ml.copy()
    df["state_crop_key"] = df["state_fips"] + "__" + df["crop"]
    keys = sorted(df["state_crop_key"].unique())
    key_to_i = {k: i for i, k in enumerate(keys)}
    df["sc_i"] = df["state_crop_key"].map(key_to_i).astype(int)

    n_sc = len(keys)
    y_obs = df["y"].to_numpy()
    r_mm = df["r"].to_numpy()
    sc_i = df["sc_i"].to_numpy()

    # ML centers array per state-crop group
    centers = (
        df[["state_fips", "crop", "log_ymax", "log_k", "state_crop_key", "sc_i"]]
        .drop_duplicates(subset=["sc_i"])
        .sort_values("sc_i")
        .reset_index(drop=True)
    )
    mu_logymax_center = centers["log_ymax"].to_numpy()
    mu_logk_center = centers["log_k"].to_numpy()

    with pm.Model() as model:
        # Non-centered: stable
        sigma_logymax = pm.HalfNormal("sigma_logymax", sigma=0.05)
        sigma_logk = pm.HalfNormal("sigma_logk", sigma=0.20)

        z_ymax = pm.Normal("z_ymax", 0.0, 1.0, shape=n_sc)
        z_k = pm.Normal("z_k", 0.0, 1.0, shape=n_sc)

        logymax = mu_logymax_center + z_ymax * sigma_logymax
        logk = mu_logk_center + z_k * sigma_logk

        ymax = pm.math.exp(logymax)
        k = pm.math.exp(logk)

        mu = ymax[sc_i] * (1 - pm.math.exp(-k[sc_i] * r_mm))

        sigma_y = pm.Exponential("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_obs)

        idata = pm.sample(
            draws=1200,
            tune=1200,
            chains=4,
            target_accept=0.95,
            random_seed=123,
            progressbar=True,
        )

    return idata


def posterior_draws_1d(idata, name):
    return idata.posterior[name].values.reshape(-1)


def loso_score(df_sy_ml, idata_sigmas, n_mc=1000):
    """
    LOSO scoring:
    For each held-out state:
      - ML-only mean prediction (deterministic using ML centers)
      - Bayes predictive distribution:
          sample sigma_logymax, sigma_logk, sigma_y from posterior,
          then sample delta_ymax, delta_k ~ Normal(0, sigma),
          compute predictive y with extra Normal noise sigma_y.
    We compute:
      - MAE (ML mean vs y_obs)
      - MAE (Bayes mean vs y_obs)
      - 90% coverage (Bayes interval contains y_obs)
    """
    df = df_sy_ml.copy()

    sig_ymax = posterior_draws_1d(idata_sigmas, "sigma_logymax")
    sig_k = posterior_draws_1d(idata_sigmas, "sigma_logk")
    sig_y = posterior_draws_1d(idata_sigmas, "sigma_y")

    rng = np.random.default_rng(123)
    states = sorted(df["state_fips"].unique())

    rows = []
    for s in states:
        test = df[df["state_fips"] == s].copy()
        if test.empty:
            continue

        # ML-only mean
        ml_ymax = np.exp(test["log_ymax"].to_numpy())
        ml_k = np.exp(test["log_k"].to_numpy())
        r = test["r"].to_numpy()
        y_true = test["y"].to_numpy()

        y_ml = sat_yield(r, ml_ymax, ml_k)

        # Bayes predictive MC
        # sample indices from posterior sigma draws
        idx = rng.integers(0, len(sig_y), size=n_mc)
        # for each observation, build predictive samples (vectorized per obs)
        # We sample deltas per observation per draw (independent).
        # (This is correct for held-out states: no data => parameter uncertainty remains.)
        sigymax_s = sig_ymax[idx]
        sigk_s = sig_k[idx]
        sigy_s = sig_y[idx]

        # shape: (n_obs, n_mc)
        delta_logymax = rng.normal(0.0, sigymax_s, size=(len(test), n_mc))
        delta_logk = rng.normal(0.0, sigk_s, size=(len(test), n_mc))

        logymax = test["log_ymax"].to_numpy()[:, None] + delta_logymax
        logk = test["log_k"].to_numpy()[:, None] + delta_logk

        ymax = np.exp(logymax)
        kk = np.exp(logk)

        mu = ymax * (1 - np.exp(-kk * r[:, None]))
        y_pred = rng.normal(mu, sigy_s[None, :], size=mu.shape)

        y_bayes_mean = y_pred.mean(axis=1)
        lo = np.quantile(y_pred, 0.05, axis=1)
        hi = np.quantile(y_pred, 0.95, axis=1)
        covered = (y_true >= lo) & (y_true <= hi)

        rows.append(
            {
                "state_fips": s,
                "n": len(test),
                "mae_ml": float(np.mean(np.abs(y_ml - y_true))),
                "mae_bayes": float(np.mean(np.abs(y_bayes_mean - y_true))),
                "cov90_bayes": float(np.mean(covered)),
                "crop_mix": ",".join(sorted(test["crop"].unique())),
            }
        )

    out = pd.DataFrame(rows).sort_values("state_fips").reset_index(drop=True)
    return out


def main():
    sy = build_state_year_table()
    ml, ml_path = load_ml_centers()

    df = sy.merge(ml, on=["state_fips", "crop"], how="inner").copy()

    print(f"Loaded ML centers from: {ml_path}")
    print("State-year obs used:", len(df))
    print("States used:", df["state_fips"].nunique())
    print("State×crop groups used:", df.groupby(["state_fips", "crop"]).ngroups)

    print("\nFitting global Bayes sigmas (this is the only expensive step)...")
    idata = fit_global_bayes_sigmas(df)

    print("\nBayes hyperparameters (global fit):")
    print(az.summary(idata, var_names=["sigma_logymax", "sigma_logk", "sigma_y"]))

    print("\nScoring LOSO (per state)...")
    res = loso_score(df, idata_sigmas=idata, n_mc=1200)

    print("\n=== LOSO summary (all states) ===")
    print("Avg MAE (ML-only):   ", round(res["mae_ml"].mean(), 3))
    print("Avg MAE (Bayes mean):", round(res["mae_bayes"].mean(), 3))
    print("Avg 90% coverage:    ", round(res["cov90_bayes"].mean(), 3))

    # By crop
    # (A state can have maize only, wheat only, or both; we’ll score rows separately by crop too.)
    by_crop = []
    for crop in sorted(df["crop"].unique()):
        df_c = df[df["crop"] == crop].copy()
        idata_c = idata  # reuse global sigmas for simplicity
        res_c = loso_score(df_c, idata_sigmas=idata_c, n_mc=1200)
        by_crop.append(
            {
                "crop": crop,
                "states": int(res_c.shape[0]),
                "mae_ml": float(res_c["mae_ml"].mean()),
                "mae_bayes": float(res_c["mae_bayes"].mean()),
                "cov90": float(res_c["cov90_bayes"].mean()),
            }
        )

    by_crop = pd.DataFrame(by_crop)
    print("\n=== LOSO by crop ===")
    print(by_crop.to_string(index=False))

    # Save per-state results for the repo
    out_path = "data/processed/loso_state_bayes_vs_ml.csv"
    res.to_csv(out_path, index=False)
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()