from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class ClimateConfig:
    processed_dir: Path = Path("data/processed")
    prism_county_year_file: str = "prism_county_year_growseason.csv"


def load_or_stub_climate_panel(cfg: ClimateConfig) -> pd.DataFrame:
    """
    Loads PRISM-derived county-year-crop growing-season precipitation, if present.

    Returns schema:
      - year (int)
      - county_fips (str)
      - crop (str)
      - precip_mm (float)
    """
    path = cfg.processed_dir / cfg.prism_county_year_file
    if not path.exists():
        return pd.DataFrame(columns=["year", "county_fips", "crop", "precip_mm"])

    df = pd.read_csv(path, dtype={"county_fips": str, "crop": str})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    out = df.rename(columns={"ppt_mm_gs": "precip_mm"})[["year", "county_fips", "crop", "precip_mm"]].copy()
    out = out.dropna(subset=["year", "precip_mm"])
    out["year"] = out["year"].astype(int)
    out["county_fips"] = out["county_fips"].astype(str).str.zfill(5)
    return out