import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def sat_curve(r, ymax, k):
    return ymax * (1 - np.exp(-k * r))


def main():
    # --- reload the same data used in bayes script ---
    panel = pd.read_csv(
        "data/processed/panel_county_year.csv",
        dtype={"state_fips": str, "county_fips": str, "crop": str},
    )
    panel = panel[panel["yield_value"] > 0].copy()

    sy = (
        panel.groupby(["state_fips", "crop", "year"], as_index=False)
        .agg(y=("yield_value", "mean"), r=("precip_mm", "mean"))
    )

    ml = pd.read_csv("data/processed/ml_state_joint_v2.csv", dtype={"state_fips": str, "crop": str})
    ml = ml[["state_fips", "crop", "log_ymax", "log_k"]].copy()

    df = sy.merge(ml, on=["state_fips", "crop"], how="inner").copy()
    df["state_crop_key"] = df["state_fips"] + "__" + df["crop"]

    # choose two groups:
    # - dense: most rows
    # - sparse: fewest rows
    counts = df["state_crop_key"].value_counts()
    dense_key = counts.index[0]
    sparse_key = counts.index[-1]

    print("Dense group:", dense_key, "n=", int(counts.loc[dense_key]))
    print("Sparse group:", sparse_key, "n=", int(counts.loc[sparse_key]))

    # --- load posterior from the last run (idata saved in memory only previously) ---
    # So: we rerun the model quickly with fewer draws but RETURN idata for plotting.
    # This is still fast at state-level and guarantees we have idata available here.
    # (If you want, later we can save idata to netcdf and load it instead.)

    # Build compact state-crop indexing
    keys = sorted(df["state_crop_key"].unique())
    key_to_i = {k: i for i, k in enumerate(keys)}
    df["sc_i"] = df["state_crop_key"].map(key_to_i).astype(int)

    n_sc = len(keys)
    y_obs = df["y"].to_numpy()
    r_mm = df["r"].to_numpy()
    sc_i = df["sc_i"].to_numpy()

    centers = ml.copy()
    centers["state_crop_key"] = centers["state_fips"] + "__" + centers["crop"]
    centers = centers[centers["state_crop_key"].isin(keys)].copy()
    centers["sc_i"] = centers["state_crop_key"].map(key_to_i).astype(int)

    mu_logymax_center = np.zeros(n_sc, dtype=float)
    mu_logk_center = np.zeros(n_sc, dtype=float)
    tmp = centers.drop_duplicates(subset=["sc_i"]).sort_values("sc_i")
    mu_logymax_center[tmp["sc_i"].to_numpy()] = tmp["log_ymax"].to_numpy()
    mu_logk_center[tmp["sc_i"].to_numpy()] = tmp["log_k"].to_numpy()

    with pm.Model() as model:
        sigma_logymax = pm.HalfNormal("sigma_logymax", sigma=0.05)
        sigma_logk = pm.HalfNormal("sigma_logk", sigma=0.20)

        z_ymax = pm.Normal("z_ymax", 0.0, 1.0, shape=n_sc)
        z_k = pm.Normal("z_k", 0.0, 1.0, shape=n_sc)

        logymax = pm.Deterministic("logymax", mu_logymax_center + z_ymax * sigma_logymax)
        logk = pm.Deterministic("logk", mu_logk_center + z_k * sigma_logk)

        ymax = pm.Deterministic("ymax", pm.math.exp(logymax))
        k = pm.Deterministic("k", pm.math.exp(logk))

        mu = ymax[sc_i] * (1 - pm.math.exp(-k[sc_i] * r_mm))

        sigma_y = pm.Exponential("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_obs)

        idata = pm.sample(draws=800, tune=800, chains=2, target_accept=0.95, random_seed=123)

    # Helper to plot one group
    def plot_group(state_crop_key: str, filename: str):
        g = df[df["state_crop_key"] == state_crop_key].copy()
        idx = key_to_i[state_crop_key]

        # ML curve
        ml_row = centers[centers["sc_i"] == idx].iloc[0]
        ml_ymax = float(np.exp(ml_row["log_ymax"]))
        ml_k = float(np.exp(ml_row["log_k"]))

        # Posterior samples for this group
        post_ymax = idata.posterior["ymax"].sel(ymax_dim_0=idx).values.reshape(-1)
        post_k = idata.posterior["k"].sel(k_dim_0=idx).values.reshape(-1)

        r_grid = np.linspace(g["r"].min(), g["r"].max(), 200)

        # Posterior predictive curve band
        curves = np.array([sat_curve(r_grid, ym, kk) for ym, kk in zip(post_ymax, post_k)])
        lo = np.quantile(curves, 0.05, axis=0)
        hi = np.quantile(curves, 0.95, axis=0)
        mid = np.mean(curves, axis=0)

        plt.figure()
        plt.scatter(g["r"], g["y"], alpha=0.6)
        plt.plot(r_grid, sat_curve(r_grid, ml_ymax, ml_k), linewidth=2, label="ML prior center")
        plt.plot(r_grid, mid, linewidth=2, label="Bayes posterior mean")
        plt.fill_between(r_grid, lo, hi, alpha=0.2, label="90% band")
        plt.xlabel("State-year mean growing-season precip (mm)")
        plt.ylabel("State-year mean yield (BU/ACRE)")
        plt.title(f"{state_crop_key} — ML vs Bayes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.show()
        print("Saved:", filename)

    plot_group(dense_key, "posterior_curve_dense.png")
    plot_group(sparse_key, "posterior_curve_sparse.png")


if __name__ == "__main__":
    main()