import numpy as np
import pandas as pd
import pymc as pm


def flatten(trace, var):
    x = trace.posterior[var].values
    return x.reshape((-1,) + x.shape[2:])


def prepare_data(n_groups=400):
    df = pd.read_csv(
        "data/processed/panel_county_year_v4.csv",
        dtype={"county_fips": str, "state_fips": str, "crop": str},
    )

    df = df[
        np.isfinite(df["precip_mm"])
        & np.isfinite(df["yield_value"])
        & np.isfinite(df["irrig_frac"])
        & (df["yield_value"] > 0)
    ].copy()

    df["county_crop"] = df["county_fips"] + "__" + df["crop"]
    keep = sorted(df["county_crop"].unique())[:n_groups]
    return df[df["county_crop"].isin(keep)].copy()


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

    group_state_idx = np.array([state_to_i[s] for s in group_info["state_fips"]])
    group_crop_idx = np.array([crop_to_i[c] for c in group_info["crop"]])

    R = df_train["precip_mm"].to_numpy(float)
    I = df_train["irrig_frac"].to_numpy(float)
    Y = df_train["yield_value"].to_numpy(float)

    n_crops = len(crop_codes)
    n_states = len(state_codes)
    n_groups = len(group_codes)

    with pm.Model() as model:
        # Crop-level priors (county-level ymax/k are centered on these)
        mu_logymax_crop = pm.Normal("mu_logymax_crop", 4.5, 1.5, shape=n_crops)
        mu_logk_crop = pm.Normal("mu_logk_crop", -6.0, 2.0, shape=n_crops)

        # State×crop baseline intercept
        tau_ymin = pm.HalfNormal("tau_ymin", 20.0)
        ymin0_statecrop = pm.Normal(
            "ymin0_statecrop",
            mu=0.0,
            sigma=tau_ymin,
            shape=(n_states, n_crops),
        )

        # Crop-level irrigation baseline slope (baseline shift, not water substitution)
        beta_irrig_crop = pm.HalfNormal("beta_irrig_crop", 200.0, shape=n_crops)

        # County random effects for ymax and k
        sigma_logymax = pm.HalfNormal("sigma_logymax", 0.6)
        sigma_logk = pm.HalfNormal("sigma_logk", 0.6)

        logymax = pm.Normal(
            "logymax",
            mu=mu_logymax_crop[group_crop_idx],
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

        # Baseline ymin for each group (state×crop intercept + crop irrig slope * irrig_frac)
        ymin0_group = ymin0_statecrop[group_state_idx, group_crop_idx]
        beta_group = beta_irrig_crop[group_crop_idx]
        ymin_group = ymin0_group + beta_group * pm.math.clip(I.mean(), 0.0, 1.0)

        # IMPORTANT:
        # Use irrigation at the observation level for baseline shift (not averaged)
        ymin_obs = ymin0_group[group_idx_obs] + beta_group[group_idx_obs] * pm.math.clip(I, 0.0, 1.0)

        mu = ymin_obs + (ymax[group_idx_obs] - ymin_obs) * (1 - pm.math.exp(-k[group_idx_obs] * R))

        sigma_y = pm.HalfNormal("sigma_y", 20.0)
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=Y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=2,
            target_accept=0.95,
            progressbar=True,
        )

    meta = {"crop_codes": crop_codes}
    return trace, meta


def predict_transfer(df_test, trace, meta, seed=0, max_draws=500):
    rng = np.random.default_rng(seed)

    crop_codes = meta["crop_codes"]
    crop_to_i = {c: i for i, c in enumerate(crop_codes)}

    df_test = df_test.copy()
    df_test["county_crop"] = df_test["county_fips"] + "__" + df_test["crop"]

    group_codes, group_idx_obs = np.unique(df_test["county_crop"], return_inverse=True)
    group_info = (
        df_test[["county_crop", "crop"]]
        .drop_duplicates()
        .sort_values("county_crop")
        .reset_index(drop=True)
    )
    group_crop_idx = np.array([crop_to_i[c] for c in group_info["crop"]])

    R = df_test["precip_mm"].to_numpy(float)
    I = df_test["irrig_frac"].to_numpy(float)
    Y = df_test["yield_value"].to_numpy(float)

    mu_logymax_crop = flatten(trace, "mu_logymax_crop")
    mu_logk_crop = flatten(trace, "mu_logk_crop")
    beta_irrig_crop = flatten(trace, "beta_irrig_crop")

    tau_ymin = flatten(trace, "tau_ymin").squeeze()
    sigma_logymax = flatten(trace, "sigma_logymax").squeeze()
    sigma_logk = flatten(trace, "sigma_logk").squeeze()
    sigma_y = flatten(trace, "sigma_y").squeeze()

    S = len(tau_ymin)
    idx = rng.choice(S, size=min(max_draws, S), replace=False)
    Sd = len(idx)

    mu_logymax_crop = mu_logymax_crop[idx]
    mu_logk_crop = mu_logk_crop[idx]
    beta_irrig_crop = beta_irrig_crop[idx]

    tau_ymin = tau_ymin[idx]
    sigma_logymax = sigma_logymax[idx]
    sigma_logk = sigma_logk[idx]
    sigma_y = sigma_y[idx]

    G = len(group_codes)
    N = len(Y)

    # Held-out: sample state×crop baseline intercept as Normal(0, tau_ymin)
    ymin0 = rng.normal(0.0, tau_ymin[:, None], size=(Sd, G))

    # County params from crop-level centers
    logymax = mu_logymax_crop[:, group_crop_idx] + rng.normal(0.0, sigma_logymax[:, None], size=(Sd, G))
    logk = mu_logk_crop[:, group_crop_idx] + rng.normal(0.0, sigma_logk[:, None], size=(Sd, G))

    ymax = np.exp(logymax)
    k = np.exp(logk)

    beta = beta_irrig_crop[:, group_crop_idx]  # (Sd, G)

    ymin_obs = ymin0[:, group_idx_obs] + beta[:, group_idx_obs] * np.clip(I[None, :], 0.0, 1.0)

    mu = ymin_obs + (ymax[:, group_idx_obs] - ymin_obs) * (1 - np.exp(-k[:, group_idx_obs] * R[None, :]))

    y_pred = mu + rng.normal(0.0, sigma_y[:, None], size=(Sd, N))

    y_mean = y_pred.mean(axis=0)
    lo = np.quantile(y_pred, 0.05, axis=0)
    hi = np.quantile(y_pred, 0.95, axis=0)

    mae = float(np.mean(np.abs(y_mean - Y)))
    cov90 = float(np.mean((Y >= lo) & (Y <= hi)))
    return mae, cov90


def main():
    df = prepare_data(n_groups=400)
    states = sorted(df["state_fips"].unique())
    test_states = states[:5]

    print("LOSO states:", test_states)

    rows = []

    for s in test_states:
        train = df[df["state_fips"] != s]
        test = df[df["state_fips"] == s]

        trace, meta = fit_model(train)
        mae, cov90 = predict_transfer(test, trace, meta)

        print(f"State {s}: MAE={mae:.2f}  cov90={cov90:.3f}")

        rows.append({"state_fips": s, "mae": mae, "cov90": cov90, "n_test": len(test)})

    res = pd.DataFrame(rows)
    out = "data/processed/loso_county_bayes_v1.csv"
    res.to_csv(out, index=False)

    print("\nSaved:", out)
    print(res)
    print("\nMean MAE:", res["mae"].mean())
    print("Mean cov90:", res["cov90"].mean())


if __name__ == "__main__":
    main()