import numpy as np
import pandas as pd
import pymc as pm


def flatten(trace, var):
    x = trace.posterior[var].values
    return x.reshape((-1,) + x.shape[2:])


def load_state_features():
    feats = pd.read_csv(
        "data/processed/state_features_from_panel.csv",
        dtype={"state_fips": str, "crop": str},
    )

    feats = feats.sort_values(["state_fips", "crop"]).copy()
    feats["crop_rank"] = (feats["crop"] != "maize").astype(int)
    feats = feats.sort_values(["state_fips", "crop_rank"]).drop_duplicates("state_fips")

    out = feats[["state_fips", "precip_mean", "yield_mean"]].copy()
    out["precip_mean"] = out["precip_mean"].astype(float)
    out["yield_mean"] = out["yield_mean"].astype(float)
    return out


def prepare_data(n_groups=800):
    df = pd.read_csv(
        "data/processed/panel_county_year_v4.csv",
        dtype={"county_fips": str, "state_fips": str, "crop": str},
    )

    needed = ["precip_mm", "yield_value", "irrig_frac", "heat_excess_sum_gs"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns in panel_county_year_v4.csv: {miss}")

    df = df[
        np.isfinite(df["precip_mm"])
        & np.isfinite(df["yield_value"])
        & np.isfinite(df["irrig_frac"])
        & np.isfinite(df["heat_excess_sum_gs"])
        & (df["yield_value"] > 0)
    ].copy()

    df["county_crop"] = df["county_fips"] + "__" + df["crop"]
    keep = sorted(df["county_crop"].unique())[:n_groups]
    df = df[df["county_crop"].isin(keep)].copy()

    sf = load_state_features()
    df = df.merge(sf, on="state_fips", how="left")

    if df["precip_mean"].isna().any():
        missing = sorted(df.loc[df["precip_mean"].isna(), "state_fips"].unique())
        raise RuntimeError(f"Missing state precip_mean for states: {missing[:20]}")

    if df["yield_mean"].isna().any():
        missing = sorted(df.loc[df["yield_mean"].isna(), "state_fips"].unique())
        raise RuntimeError(f"Missing state yield_mean for states: {missing[:20]}")

    return df


def fit_model(df_train, draws=400, tune=600):
    df_train = df_train.copy()
    df_train["county_crop"] = df_train["county_fips"] + "__" + df_train["crop"]

    crop_codes = np.sort(df_train["crop"].unique())
    state_codes = np.sort(df_train["state_fips"].unique())

    crop_to_i = {c: i for i, c in enumerate(crop_codes)}
    state_to_i = {s: i for i, s in enumerate(state_codes)}

    group_codes, group_idx_obs = np.unique(df_train["county_crop"], return_inverse=True)

    group_info = (
        df_train[["county_crop", "state_fips", "crop"]]
        .drop_duplicates()
        .sort_values("county_crop")
        .reset_index(drop=True)
    )

    group_state_idx = np.array([state_to_i[s] for s in group_info["state_fips"]], dtype=int)
    group_crop_idx = np.array([crop_to_i[c] for c in group_info["crop"]], dtype=int)

    state_feats = (
        df_train[["state_fips", "precip_mean", "yield_mean"]]
        .drop_duplicates()
        .set_index("state_fips")
    )

    x_precip = np.array([float(state_feats.loc[s, "precip_mean"]) for s in state_codes], dtype=float)
    x_yield = np.array([float(state_feats.loc[s, "yield_mean"]) for s in state_codes], dtype=float)

    precip_mu = float(x_precip.mean())
    precip_sd = float(x_precip.std(ddof=0))
    if precip_sd <= 0:
        raise RuntimeError("state precip_mean has zero std; cannot standardize.")
    x_precip_z = (x_precip - precip_mu) / precip_sd

    yield_mu = float(x_yield.mean())
    yield_sd = float(x_yield.std(ddof=0))
    if yield_sd <= 0:
        raise RuntimeError("state yield_mean has zero std; cannot standardize.")
    x_yield_z = (x_yield - yield_mu) / yield_sd

    R = df_train["precip_mm"].to_numpy(dtype=float)
    I = np.clip(df_train["irrig_frac"].to_numpy(dtype=float), 0.0, 1.0)
    H = df_train["heat_excess_sum_gs"].to_numpy(dtype=float)
    Y = df_train["yield_value"].to_numpy(dtype=float)

    crop_idx_obs = np.array([crop_to_i[c] for c in df_train["crop"]], dtype=int)

    n_crops = len(crop_codes)
    n_states = len(state_codes)
    n_groups = len(group_codes)

    with pm.Model() as model:
        mu_logymax_crop = pm.Normal("mu_logymax_crop", 4.5, 1.5, shape=n_crops)
        mu_logk_crop = pm.Normal("mu_logk_crop", -6.0, 2.0, shape=n_crops)

        gamma_ymax_crop = pm.Normal("gamma_ymax_crop", 0.0, 0.5, shape=n_crops)

        alpha0_crop = pm.Normal("alpha0_crop", 0.0, 40.0, shape=n_crops)
        alphaP_crop = pm.Normal("alphaP_crop", 0.0, 40.0, shape=n_crops)

        tau_ymin = pm.HalfNormal("tau_ymin", 20.0)
        u_statecrop = pm.Normal("u_statecrop", 0.0, tau_ymin, shape=(n_states, n_crops))

        ymin0_statecrop = (
            alpha0_crop[None, :]
            + alphaP_crop[None, :] * x_precip_z[:, None]
            + u_statecrop
        )

        beta_irrig_crop = pm.HalfNormal("beta_irrig_crop", 200.0, shape=n_crops)

        # NEW: crop-level heat damage slope
        beta_heat_crop = pm.HalfNormal("beta_heat_crop", 5.0, shape=n_crops)

        sigma_logymax = pm.HalfNormal("sigma_logymax", 0.6)
        sigma_logk = pm.HalfNormal("sigma_logk", 0.6)

        mu_logymax_group = (
            mu_logymax_crop[group_crop_idx]
            + gamma_ymax_crop[group_crop_idx] * x_yield_z[group_state_idx]
        )

        logymax = pm.Normal(
            "logymax",
            mu=mu_logymax_group,
            sigma=sigma_logymax,
            shape=n_groups,
        )

        logk = pm.Normal(
            "logk",
            mu=mu_logk_crop[group_crop_idx],
            sigma=sigma_logk,
            shape=n_groups,
        )

        ymax = pm.math.exp(logymax)
        k = pm.math.exp(logk)

        ymin0_group = ymin0_statecrop[group_state_idx, group_crop_idx]
        beta_group = beta_irrig_crop[group_crop_idx]
        ymin_obs = ymin0_group[group_idx_obs] + beta_group[group_idx_obs] * I

        heat_damage = beta_heat_crop[crop_idx_obs] * H

        mu = (
            ymin_obs
            + (ymax[group_idx_obs] - ymin_obs) * (1 - pm.math.exp(-k[group_idx_obs] * R))
            - heat_damage
        )

        sigma_y = pm.HalfNormal("sigma_y", 20.0)
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=Y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=4,
            target_accept=0.95,
            progressbar=True,
        )

    meta = {
        "crop_codes": crop_codes,
        "precip_mu": precip_mu,
        "precip_sd": precip_sd,
        "yield_mu": yield_mu,
        "yield_sd": yield_sd,
    }
    return trace, meta


def predict_transfer(df_test, trace, meta, seed=0, max_draws=500):
    rng = np.random.default_rng(seed)

    crop_codes = meta["crop_codes"]
    crop_to_i = {c: i for i, c in enumerate(crop_codes)}

    precip_mu = float(meta["precip_mu"])
    precip_sd = float(meta["precip_sd"])
    yield_mu = float(meta["yield_mu"])
    yield_sd = float(meta["yield_sd"])

    x_precip_z_test = float((df_test["precip_mean"].iloc[0] - precip_mu) / precip_sd)
    x_yield_z_test = float((df_test["yield_mean"].iloc[0] - yield_mu) / yield_sd)

    df_test = df_test.copy()
    df_test["county_crop"] = df_test["county_fips"] + "__" + df_test["crop"]

    group_codes, group_idx_obs = np.unique(df_test["county_crop"], return_inverse=True)
    group_info = (
        df_test[["county_crop", "crop"]]
        .drop_duplicates()
        .sort_values("county_crop")
        .reset_index(drop=True)
    )
    group_crop_idx = np.array([crop_to_i[c] for c in group_info["crop"]], dtype=int)

    R = df_test["precip_mm"].to_numpy(dtype=float)
    I = np.clip(df_test["irrig_frac"].to_numpy(dtype=float), 0.0, 1.0)
    H = df_test["heat_excess_sum_gs"].to_numpy(dtype=float)
    Y = df_test["yield_value"].to_numpy(dtype=float)

    crop_idx_obs = np.array([crop_to_i[c] for c in df_test["crop"]], dtype=int)

    alpha0_crop = flatten(trace, "alpha0_crop")
    alphaP_crop = flatten(trace, "alphaP_crop")
    beta_irrig_crop = flatten(trace, "beta_irrig_crop")
    beta_heat_crop = flatten(trace, "beta_heat_crop")
    gamma_ymax_crop = flatten(trace, "gamma_ymax_crop")

    mu_logymax_crop = flatten(trace, "mu_logymax_crop")
    mu_logk_crop = flatten(trace, "mu_logk_crop")

    tau_ymin = flatten(trace, "tau_ymin").squeeze()
    sigma_logymax = flatten(trace, "sigma_logymax").squeeze()
    sigma_logk = flatten(trace, "sigma_logk").squeeze()
    sigma_y = flatten(trace, "sigma_y").squeeze()

    S = len(tau_ymin)
    idx = rng.choice(S, size=min(max_draws, S), replace=False)
    Sd = len(idx)

    alpha0_crop = alpha0_crop[idx]
    alphaP_crop = alphaP_crop[idx]
    beta_irrig_crop = beta_irrig_crop[idx]
    beta_heat_crop = beta_heat_crop[idx]
    gamma_ymax_crop = gamma_ymax_crop[idx]
    mu_logymax_crop = mu_logymax_crop[idx]
    mu_logk_crop = mu_logk_crop[idx]
    tau_ymin = tau_ymin[idx]
    sigma_logymax = sigma_logymax[idx]
    sigma_logk = sigma_logk[idx]
    sigma_y = sigma_y[idx]

    G = len(group_codes)
    N = len(Y)

    u = rng.normal(0.0, tau_ymin[:, None], size=(Sd, G))
    ymin0 = (alpha0_crop[:, group_crop_idx] + alphaP_crop[:, group_crop_idx] * x_precip_z_test) + u

    mu_logymax_group = (
        mu_logymax_crop[:, group_crop_idx]
        + gamma_ymax_crop[:, group_crop_idx] * x_yield_z_test
    )
    logymax = mu_logymax_group + rng.normal(0.0, sigma_logymax[:, None], size=(Sd, G))
    logk = mu_logk_crop[:, group_crop_idx] + rng.normal(0.0, sigma_logk[:, None], size=(Sd, G))

    ymax = np.exp(logymax)
    k = np.exp(logk)

    beta = beta_irrig_crop[:, group_crop_idx]
    ymin_obs = ymin0[:, group_idx_obs] + beta[:, group_idx_obs] * I[None, :]

    heat_damage = beta_heat_crop[:, crop_idx_obs] * H[None, :]

    mu = (
        ymin_obs
        + (ymax[:, group_idx_obs] - ymin_obs) * (1 - np.exp(-k[:, group_idx_obs] * R[None, :]))
        - heat_damage
    )

    y_pred = mu + rng.normal(0.0, sigma_y[:, None], size=(Sd, N))

    y_mean = y_pred.mean(axis=0)
    lo = np.quantile(y_pred, 0.05, axis=0)
    hi = np.quantile(y_pred, 0.95, axis=0)

    mae = float(np.mean(np.abs(y_mean - Y)))
    cov90 = float(np.mean((Y >= lo) & (Y <= hi)))
    return mae, cov90


def main():
    df = prepare_data(n_groups=800)
    states = sorted(df["state_fips"].unique())
    test_states = states

    print("LOSO states:", test_states)

    rows = []
    for s in test_states:
        train = df[df["state_fips"] != s]
        test = df[df["state_fips"] == s].copy()
        if len(test) == 0:
            continue

        trace, meta = fit_model(train)
        mae, cov90 = predict_transfer(test, trace, meta)

        print(f"State {s}: MAE={mae:.2f}  cov90={cov90:.3f}")
        rows.append({"state_fips": s, "mae": mae, "cov90": cov90, "n_test": len(test)})

    res = pd.DataFrame(rows).sort_values("state_fips").reset_index(drop=True)
    out = "data/processed/loso_county_bayes_v5_heat.csv"
    res.to_csv(out, index=False)

    print("\nSaved:", out)
    print(res.head(20))
    print("\nMean MAE:", res["mae"].mean())
    print("Mean cov90:", res["cov90"].mean())


if __name__ == "__main__":
    main()