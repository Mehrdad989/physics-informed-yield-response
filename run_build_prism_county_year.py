import socket
import zipfile
from pathlib import Path

import pandas as pd

from piyrm.data.sources.prism import PrismConfig, download_prism_month
from piyrm.data.sources.tiger_counties import TigerCountiesConfig, load_tiger_counties
from piyrm.data.geo.zonal import zonal_mean

# Global default timeout for urllib/network ops
socket.setdefaulttimeout(120)


def find_raster_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        tifs = [n for n in names if n.lower().endswith(".tif")]
        if tifs:
            return tifs[0]
        bils = [n for n in names if n.lower().endswith(".bil")]
        if bils:
            return bils[0]
    raise RuntimeError(f"No .tif or .bil found in {zip_path}")


def extract_member(zip_path: Path, member: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(member, path=out_dir)
    return out_dir / member


def county_month_cache_path(year: int, month: int) -> Path:
    return Path("data/processed/prism_county_month") / f"{year}_{month:02d}.csv"


def prism_county_month_mean_cached(year: int, month: int, counties_gdf) -> pd.DataFrame:
    """
    Returns county_fips + ppt_mm for a given month.
    Uses cached CSV if present.
    If a month fails, writes a .FAILED.txt marker and returns empty.
    """
    cache = county_month_cache_path(year, month)
    cache.parent.mkdir(parents=True, exist_ok=True)

    fail_marker = cache.with_suffix(".FAILED.txt")
    if fail_marker.exists():
        return pd.DataFrame(columns=["county_fips", "ppt_mm"])

    if cache.exists():
        df = pd.read_csv(cache, dtype={"county_fips": str})
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        df["ppt_mm"] = pd.to_numeric(df["ppt_mm"], errors="coerce")
        return df

    prism_cfg = PrismConfig()  # uses retries/timeouts from prism.py

    try:
        zip_path = download_prism_month(prism_cfg, year, month)
        member = find_raster_in_zip(zip_path)
        raster_path = extract_member(zip_path, member, Path("data/raw/prism/extracted"))

        stats = zonal_mean(polygons=counties_gdf, raster_path=str(raster_path), id_col="county_fips")
        out = stats.rename(columns={"raster_mean": "ppt_mm"}).copy()

        out["county_fips"] = out["county_fips"].astype(str).str.zfill(5)
        out["ppt_mm"] = pd.to_numeric(out["ppt_mm"], errors="coerce")

        out.to_csv(cache, index=False)
        return out

    except Exception as e:
        fail_marker.write_text(str(e), encoding="utf-8")
        print(f"WARNING: failed PRISM {year}-{month:02d}: {e}")
        return pd.DataFrame(columns=["county_fips", "ppt_mm"])


def main():
    # Match your NASS range
    start_year = 2000
    end_year = 2023

    # Seasons (edit later if you want)
    maize_months = [4, 5, 6, 7, 8, 9]         # Apr–Sep
    wheat_prev_months = [10, 11, 12]          # Oct–Dec of (year-1)
    wheat_curr_months = [1, 2, 3, 4, 5, 6]    # Jan–Jun of (year)

    counties = load_tiger_counties(TigerCountiesConfig(year=2023))

    rows = []

    # --- maize: within-year ---
    for y in range(start_year, end_year + 1):
        for m in maize_months:
            dfm = prism_county_month_mean_cached(y, m, counties)
            if len(dfm) == 0:
                continue
            dfm["year"] = y
            dfm["month"] = m
            dfm["crop"] = "maize"
            rows.append(dfm)

    # --- wheat: harvest year y uses Oct–Dec (y-1) + Jan–Jun (y) ---
    for y in range(start_year, end_year + 1):
        for m in wheat_prev_months:
            dfm = prism_county_month_mean_cached(y - 1, m, counties)
            if len(dfm) == 0:
                continue
            dfm["year"] = y
            dfm["month"] = m
            dfm["crop"] = "wheat"
            rows.append(dfm)

        for m in wheat_curr_months:
            dfm = prism_county_month_mean_cached(y, m, counties)
            if len(dfm) == 0:
                continue
            dfm["year"] = y
            dfm["month"] = m
            dfm["crop"] = "wheat"
            rows.append(dfm)

    if not rows:
        raise RuntimeError("No PRISM county-month data produced. Check downloads / FAILED markers.")

    df = pd.concat(rows, ignore_index=True)

    # Aggregate to county-year-crop growing season total precip
    county_year = (
        df.groupby(["county_fips", "year", "crop"], as_index=False)["ppt_mm"]
        .sum()
        .rename(columns={"ppt_mm": "ppt_mm_gs"})
    )

    out_path = Path("data/processed/prism_county_year_growseason.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    county_year.to_csv(out_path, index=False)

    print(f"Wrote: {out_path} (rows={len(county_year)})")
    print(county_year.head())


if __name__ == "__main__":
    main()