import os
import numpy as np
import matplotlib.pyplot as plt

from piyrm.features import generate_region_features
from piyrm.synthetic_crops import generate_synthetic_region_crop_yield, CROP_NAMES
from piyrm.fit import fit_saturating_by_crop_region
from piyrm.ml import fit_linear_multioutput
from piyrm.bayes_crops import fit_crosscrop_saturating_bayes_with_ml_priors


def curve_saturating(rain_mm: np.ndarray, ymax: np.ndarray, k: np.ndarray, y_min: float) -> np.ndarray:
    return y_min + (ymax - y_min) * (1.0 - np.exp(-k * rain_mm))


def main():
    out_dir = os.path.join("docs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Same setup as pipeline (so results are comparable)
    n_regions = 30
    n_crops = 2
    y_min_fixed = 0.5

    sparse_regions = set(range(24, 30))
    dense_n = 120
    sparse_n = 30

    feat = generate_region_features(seed=0, n_regions=n_regions)

    # Build variable-n dataset by concatenating per-region chunks
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

    # Stage 1 fits
    fits = fit_saturating_by_crop_region(crop_id=crop_id, region=region, rain_mm=rain, yield_obs=y_obs)
    ymax_hat = np.full((n_crops, n_regions), np.nan, dtype=float)
    k_hat = np.full((n_crops, n_regions), np.nan, dtype=float)
    for fr in fits:
        ymax_hat[fr.crop_id, fr.region] = fr.ymax
        k_hat[fr.crop_id, fr.region] = fr.k

    # Stage 2 ML priors (trained on dense regions only)
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

    # Stage 3 Bayes with ML priors (keep modest for speed)
    res = fit_crosscrop_saturating_bayes_with_ml_priors(
        crop_id=crop_id,
        region=region,
        rain_mm=rain,
        yield_obs=y_obs,
        ymax_ml=ymax_ml,
        k_ml=k_ml,
        y_min_fixed=y_min_fixed,
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=123,
        progressbar=True,
    )

    idata = res.idata
    n_regions = int(region.max()) + 1

    # Pick one dense region and one sparse region to visualize
    dense_region = 5
    sparse_region = 28

    rain_grid = np.linspace(0.0, 800.0, 200)

    # Posterior arrays are shape (chain, draw, n_cr)
    # n_cr = n_crops*n_regions; index = crop_id*n_regions + region
    ymax_draws = idata.posterior["ymax"].stack(sample=("chain", "draw")).values  # (n_cr, samples)
    k_draws = idata.posterior["k"].stack(sample=("chain", "draw")).values        # (n_cr, samples)

    # Make two figures: one for dense, one for sparse. Each includes both crops.
    for r0, label in [(dense_region, "dense"), (sparse_region, "sparse")]:
        plt.figure()
        for c in range(n_crops):
            cr = c * n_regions + r0

            # posterior curves: compute quantiles across samples
            y_curves = curve_saturating(
                rain_grid[None, :],
                ymax_draws[cr, :][:, None],
                k_draws[cr, :][:, None],
                y_min_fixed,
            )  # (samples, grid)

            lo = np.quantile(y_curves, 0.05, axis=0)
            hi = np.quantile(y_curves, 0.95, axis=0)
            mid = np.quantile(y_curves, 0.50, axis=0)

            plt.plot(rain_grid, mid, label=f"{CROP_NAMES[c]} posterior median")
            plt.fill_between(rain_grid, lo, hi, alpha=0.2)

            # overlay observed points for this crop-region
            m = (region == r0) & (crop_id == c)
            plt.scatter(rain[m], y_obs[m], s=12, alpha=0.4, label=f"{CROP_NAMES[c]} obs")

        plt.xlabel("Seasonal rainfall (mm)")
        plt.ylabel("Yield (t/ha)")
        plt.title(f"Posterior response curves with 90% bands ({label} region {r0})")
        plt.legend()
        plt.tight_layout()

        path = os.path.join(out_dir, f"posterior_curves_{label}_region_{r0}.png")
        plt.savefig(path, dpi=180)
        plt.close()

    print(f"Saved figures to: {out_dir}")
    print(f"- posterior_curves_dense_region_{dense_region}.png")
    print(f"- posterior_curves_sparse_region_{sparse_region}.png")


if __name__ == "__main__":
    main()