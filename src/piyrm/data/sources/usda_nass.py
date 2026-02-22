from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class NassConfig:
    """
    Placeholder config for USDA NASS data assembly.

    Later we’ll support:
    - downloading from NASS QuickStats API
    - caching raw files
    - selecting crops, years, states, aggregation level
    """
    raw_dir: Path = Path("data/raw/usda_nass")
    processed_dir: Path = Path("data/processed")
    crops: tuple[str, ...] = ("WHEAT", "CORN")  # NASS names
    unit: str = "BU / ACRE"


def load_or_stub_nass_yield(cfg: NassConfig) -> pd.DataFrame:
    """
    For now: return an empty/stub dataframe with the target schema.

    Columns (planned):
      - year (int)
      - state_fips (str)
      - county_fips (str)
      - crop (str)  # wheat/maize naming standardized later
      - yield_value (float)
      - yield_unit (str)
    """
    # Stub with correct columns so downstream code can be written now.
    cols = ["year", "state_fips", "county_fips", "crop", "yield_value", "yield_unit"]
    return pd.DataFrame(columns=cols)