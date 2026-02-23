from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class NassConfig:
    raw_dir: Path = Path("data/raw/usda_nass")
    processed_dir: Path = Path("data/processed")
    crops: tuple[str, ...] = ("CORN", "WHEAT")
    preferred_unit: str = "BU / ACRE"
    start_year: int = 2000
    end_year: int = 2023


def _clean_quickstats_yield_table(df_raw: pd.DataFrame, commodity: str, cfg: NassConfig) -> pd.DataFrame:
    if df_raw.empty:
        cols = ["year", "state_fips", "county_fips", "crop", "yield_value", "yield_unit"]
        return pd.DataFrame(columns=cols)

    df = df_raw.copy()

    # Local filters (safe; avoids API 400)
    if "statisticcat_desc" in df.columns:
        df = df[df["statisticcat_desc"].astype(str).str.upper() == "YIELD"]

    if "unit_desc" in df.columns:
        df = df[df["unit_desc"].astype(str).str.upper() == cfg.preferred_unit.upper()]

    # Corn: prefer grain if those columns exist
    if commodity == "CORN":
        if "class_desc" in df.columns:
            m = df["class_desc"].astype(str).str.upper().eq("GRAIN")
            if m.any():
                df = df[m]
        if "util_practice_desc" in df.columns:
            m = df["util_practice_desc"].astype(str).str.upper().eq("GRAIN")
            if m.any():
                df = df[m]

    required = {"year", "Value", "state_fips_code", "county_code"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"QuickStats response missing expected columns: {sorted(missing)}. "
            f"Available columns include: {list(df.columns)[:30]} ..."
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    val = df["Value"].astype(str).str.replace(",", "", regex=False)
    df["yield_value"] = pd.to_numeric(val, errors="coerce")

    state = df["state_fips_code"].astype(str).str.zfill(2)
    county = df["county_code"].astype(str).str.zfill(3)

    out = pd.DataFrame(
        {
            "year": df["year"],
            "state_fips": state,
            "county_fips": state + county,
            "yield_value": df["yield_value"],
        }
    )

    crop_map = {"CORN": "maize", "WHEAT": "wheat"}
    out["crop"] = crop_map.get(commodity, commodity.lower())
    out["yield_unit"] = cfg.preferred_unit

    out = out.dropna(subset=["year", "yield_value"])
    out["year"] = out["year"].astype(int)

    # Drop county "000" (state totals)
    out = out[~out["county_fips"].astype(str).str.endswith("000")].copy()

    return out.reset_index(drop=True)


def load_or_stub_nass_yield(cfg: NassConfig) -> pd.DataFrame:
    import os
    from piyrm.data.sources.nass_quickstats import QuickStatsConfig, quickstats_county_rows_range_chunked, save_raw_csv

    api_key = os.environ.get("NASS_QUICKSTATS_KEY", "").strip()
    if not api_key:
        cols = ["year", "state_fips", "county_fips", "crop", "yield_value", "yield_unit"]
        return pd.DataFrame(columns=cols)

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    qs_cfg = QuickStatsConfig(raw_dir=cfg.raw_dir)

    frames: list[pd.DataFrame] = []

    for commodity in cfg.crops:
        # We keep request broad; filter locally later
        df_raw = quickstats_county_rows_range_chunked(
            cfg=qs_cfg,
            commodity_desc=commodity,
            start_year=cfg.start_year,
            end_year=cfg.end_year,
            agg_level_desc="COUNTY",
            extra_params={"statisticcat_desc": "YIELD"},
            save_each_chunk_csv=True,
        )
        # Also save one combined raw file for convenience
        combined_path = cfg.raw_dir / f"quickstats_{commodity.lower()}_county_{cfg.start_year}_{cfg.end_year}_combined.csv"

        if combined_path.exists():
            df_raw = pd.read_csv(combined_path)
        else:
            df_raw = quickstats_county_rows_range_chunked(
                cfg=qs_cfg,
                commodity_desc=commodity,
                start_year=cfg.start_year,
                end_year=cfg.end_year,
                agg_level_desc="COUNTY",
                extra_params={"statisticcat_desc": "YIELD"},
                save_each_chunk_csv=True,
            )
            save_raw_csv(df_raw, combined_path)

        df_clean = _clean_quickstats_yield_table(df_raw, commodity=commodity, cfg=cfg)
        frames.append(df_clean)

    if not frames:
        cols = ["year", "state_fips", "county_fips", "crop", "yield_value", "yield_unit"]
        return pd.DataFrame(columns=cols)

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)