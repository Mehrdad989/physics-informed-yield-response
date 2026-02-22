from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class ClimateConfig:
    """
    Placeholder config for climate aggregation (PRISM/Daymet/ERA5).

    Later we’ll implement:
    - download or read gridded data
    - aggregate to county × year × crop growing season windows
    """
    raw_dir: Path = Path("data/raw/climate")
    processed_dir: Path = Path("data/processed")


def load_or_stub_climate_panel(cfg: ClimateConfig) -> pd.DataFrame:
    """
    Return an empty/stub dataframe with planned climate schema.

    Columns (planned):
      - year (int)
      - state_fips (str)
      - county_fips (str)
      - precip_mm (float)
      - gdd (float)
      - heat_days (float)
      - drought_index (float)
    """
    cols = ["year", "state_fips", "county_fips", "precip_mm", "gdd", "heat_days", "drought_index"]
    return pd.DataFrame(columns=cols)