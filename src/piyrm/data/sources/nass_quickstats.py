from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


@dataclass(frozen=True)
class QuickStatsConfig:
    api_key_env: str = "NASS_QUICKSTATS_KEY"
    raw_dir: Path = Path("data/raw/usda_nass")
    timeout_s: int = 60


def _get_api_key(cfg: QuickStatsConfig) -> str:
    key = os.environ.get(cfg.api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"Missing USDA NASS QuickStats API key. Set env var {cfg.api_key_env}.")
    return key


def quickstats_get(*, cfg: QuickStatsConfig, params: dict[str, Any]) -> pd.DataFrame:
    api_key = _get_api_key(cfg)
    url = "https://quickstats.nass.usda.gov/api/api_GET/"
    full_params = {"key": api_key, "format": "JSON", **params}

    r = requests.get(url, params=full_params, timeout=cfg.timeout_s)

    # Let caller decide how to handle 413 specifically
    if r.status_code == 413:
        raise RuntimeError(f"QuickStats 413 Payload Too Large\nURL: {r.url}")

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"QuickStats HTTP error: {e}\nURL: {r.url}") from e

    payload = r.json()
    if "data" not in payload:
        raise RuntimeError(f"Unexpected response keys={list(payload.keys())}")

    return pd.DataFrame(payload["data"])


def quickstats_county_rows_by_year(
    *,
    cfg: QuickStatsConfig,
    commodity_desc: str,
    year: int,
    agg_level_desc: str = "COUNTY",
    extra_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "sector_desc": "CROPS",
        "commodity_desc": commodity_desc,
        "agg_level_desc": agg_level_desc,
        "year": str(year),
    }
    if extra_params:
        params.update(extra_params)

    return quickstats_get(cfg=cfg, params=params)


def quickstats_county_rows_by_year_state(
    *,
    cfg: QuickStatsConfig,
    commodity_desc: str,
    year: int,
    state_fips_code: str,
    agg_level_desc: str = "COUNTY",
    extra_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "sector_desc": "CROPS",
        "commodity_desc": commodity_desc,
        "agg_level_desc": agg_level_desc,
        "year": str(year),
        "state_fips_code": str(state_fips_code).zfill(2),
    }
    if extra_params:
        params.update(extra_params)

    return quickstats_get(cfg=cfg, params=params)


def quickstats_county_rows_range_chunked(
    *,
    cfg: QuickStatsConfig,
    commodity_desc: str,
    start_year: int,
    end_year: int,
    agg_level_desc: str = "COUNTY",
    extra_params: dict[str, Any] | None = None,
    save_each_chunk_csv: bool = True,
    # all state fips codes 01..56 excluding gaps like 03, 07, etc is OK; empty states just return 0 rows
    state_fips_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Robust downloader:
    - Try year-only first
    - If 413, fall back to year×state chunks
    """
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    if state_fips_codes is None:
        state_fips_codes = [f"{i:02d}" for i in range(1, 57)]  # 01..56

    frames: list[pd.DataFrame] = []

    for y in range(int(start_year), int(end_year) + 1):
        try:
            df_y = quickstats_county_rows_by_year(
                cfg=cfg,
                commodity_desc=commodity_desc,
                year=y,
                agg_level_desc=agg_level_desc,
                extra_params=extra_params,
            )
            if save_each_chunk_csv:
                out = cfg.raw_dir / f"quickstats_{commodity_desc.lower()}_{agg_level_desc.lower()}_{y}.csv"
                df_y.to_csv(out, index=False)
            frames.append(df_y)
            continue

        except RuntimeError as e:
            # Only fall back on 413; rethrow anything else
            if "413" not in str(e):
                raise

        # Fallback: split the year by state
        for st in state_fips_codes:
            df_ys = quickstats_county_rows_by_year_state(
                cfg=cfg,
                commodity_desc=commodity_desc,
                year=y,
                state_fips_code=st,
                agg_level_desc=agg_level_desc,
                extra_params=extra_params,
            )
            if save_each_chunk_csv:
                out = cfg.raw_dir / f"quickstats_{commodity_desc.lower()}_{agg_level_desc.lower()}_{y}_st{st}.csv"
                df_ys.to_csv(out, index=False)
            frames.append(df_ys)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def save_raw_csv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path