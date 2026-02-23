import zipfile
from pathlib import Path

import pandas as pd

from piyrm.data.sources.prism import PrismConfig, download_prism_month
from piyrm.data.sources.tiger_counties import TigerCountiesConfig, load_tiger_counties
from piyrm.data.geo.zonal import zonal_mean


def find_raster_in_zip(zip_path: Path) -> str:
    """
    Return a path-like string to the first GeoTIFF we find in the zip.
    PRISM zips typically contain .bil or .tif; we prefer .tif if present.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        tifs = [n for n in names if n.lower().endswith(".tif")]
        if tifs:
            return tifs[0]
        bils = [n for n in names if n.lower().endswith(".bil")]
        if bils:
            return bils[0]
    raise RuntimeError("No .tif or .bil found in PRISM zip")


def extract_member(zip_path: Path, member: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(member, path=out_dir)
    return out_dir / member


def main():
    year, month = 2020, 6

    # 1) Download PRISM month
    prism_cfg = PrismConfig()
    zip_path = download_prism_month(prism_cfg, year, month)

    # 2) Extract raster
    member = find_raster_in_zip(zip_path)
    extracted_dir = Path("data/raw/prism/extracted")
    raster_path = extract_member(zip_path, member, extracted_dir)

    # 3) Load counties
    counties = load_tiger_counties(TigerCountiesConfig(year=2023))
    # TIGER is in lon/lat; zonal_mean will reproject to raster CRS

    # 4) Zonal mean precip
    stats = zonal_mean(polygons=counties, raster_path=str(raster_path), id_col="county_fips")
    out = stats.rename(columns={"raster_mean": "ppt_mm"}).copy()
    out["year"] = year
    out["month"] = month

    # 5) Save processed
    out_path = Path(f"data/processed/prism_county_month_{year}_{month:02d}.csv")
    out.to_csv(out_path, index=False)

    print(f"Wrote: {out_path} (rows={len(out)})")
    print(out.head())


if __name__ == "__main__":
    main()