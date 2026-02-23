from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
import urllib.request

import geopandas as gpd


@dataclass(frozen=True)
class TigerCountiesConfig:
    """
    TIGER/Line counties for a specific vintage (year).
    """
    year: int = 2023
    raw_dir: Path = Path("data/raw/tiger")


def download_tiger_counties(cfg: TigerCountiesConfig) -> Path:
    """
    Downloads TIGER/Line counties ZIP and returns the ZIP path.
    """
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cfg.raw_dir / f"tl_{cfg.year}_us_county.zip"

    if zip_path.exists():
        return zip_path

    url = f"https://www2.census.gov/geo/tiger/TIGER{cfg.year}/COUNTY/tl_{cfg.year}_us_county.zip"
    urllib.request.urlretrieve(url, zip_path)
    return zip_path


def load_tiger_counties(cfg: TigerCountiesConfig) -> gpd.GeoDataFrame:
    """
    Loads counties as GeoDataFrame with GEOID (5-digit county FIPS).
    """
    zip_path = download_tiger_counties(cfg)

    # geopandas can read zipped shapefiles directly
    gdf = gpd.read_file(zip_path)

    # Standardize key field
    if "GEOID" not in gdf.columns:
        raise RuntimeError("Expected GEOID column in TIGER county shapefile")
    gdf = gdf[["GEOID", "geometry"]].rename(columns={"GEOID": "county_fips"})
    return gdf