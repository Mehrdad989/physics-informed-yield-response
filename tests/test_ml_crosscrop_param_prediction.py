import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from piyrm.features import generate_region_features
from piyrm.fit import fit_saturating_by_crop_region
from piyrm.synthetic_crops import generate_synthetic_region_crop_yield


def test_crosscrop_ml_predicts_params_with_region_holdout():
    n_regions = 60
    feat = generate_region_features(seed=0, n_regions=n_regions)

    d = generate_synthetic_region_crop_yield(
        seed=1,
        n_regions=n_regions,
        n_per_region_crop=250,
        sigma_y=0.8,
        region_features=feat,
    )
    df = pd.DataFrame(d)

    fits = fit_saturating_by_crop_region(
        crop_id=df["crop_id"].to_numpy(),
        region=df["region"].to_numpy(),
        rain_mm=df["rain_mm"].to_numpy(),
        yield_obs=df["yield_obs"].to_numpy(),
    )

    crop_ids = np.array([fr.crop_id for fr in fits], dtype=int)
    regions = np.array([fr.region for fr in fits], dtype=int)

    ymax_hat = np.array([fr.ymax for fr in fits], dtype=float)
    k_hat = np.array([fr.k for fr in fits], dtype=float)

    Y = np.column_stack([ymax_hat, k_hat])

    # Build per-row X = region features + crop_id
    aridity = feat["aridity_index"][regions]
    soil = feat["soil_whc"][regions]
    heat = feat["heat_stress"][regions]
    X = np.column_stack([aridity, soil, heat, crop_ids])

    # Deterministic region holdout split
    train_regions = set(range(0, 48))  # 80% train
    test_regions = set(range(48, 60))  # 20% test
    train_mask = np.array([r in train_regions for r in regions], dtype=bool)
    test_mask = ~train_mask

    model = LinearRegression()
    model.fit(X[train_mask], Y[train_mask])
    pred = model.predict(X[test_mask])

    r2_ymax = r2_score(Y[test_mask, 0], pred[:, 0])
    r2_k = r2_score(Y[test_mask, 1], pred[:, 1])

    assert r2_ymax > 0.90
    assert r2_k > 0.85