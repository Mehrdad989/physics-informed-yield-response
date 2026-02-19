import numpy as np

from piyrm.fit import fit_hump_by_region
from piyrm.synthetic import generate_synthetic_region_yield


def test_fit_recovers_region_params_reasonably_well():
    d = generate_synthetic_region_yield(
        seed=123,
        n_regions=3,
        n_per_region=600,
        sigma_y=0.4,
        include_true_params=True,
    )

    fits = fit_hump_by_region(
        region=d["region"],
        rain_mm=d["rain_mm"],
        yield_obs=d["yield_obs"],
    )

    # Check recovery for each region (tolerances are intentionally modest)
    for fr in fits:
        mask = d["region"] == fr.region
        true_ymax = float(d["ymax_true"][mask][0])
        true_ropt = float(d["r_opt_true"][mask][0])
        true_width = float(d["width_true"][mask][0])

        assert np.isfinite(fr.ymax)
        assert np.isfinite(fr.r_opt_mm)
        assert np.isfinite(fr.width_mm)

        # Recovery tolerances: should usually pass for this synthetic setup
        assert abs(fr.r_opt_mm - true_ropt) < 20.0
        assert abs(fr.ymax - true_ymax) < 1.0
        assert abs(fr.width_mm - true_width) < 40.0
