from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request


@dataclass(frozen=True)
class PrismConfig:
    raw_dir: Path = Path("data/raw/prism")
    region: str = "us"         # "us" is what we want
    resolution: str = "4km"    # "4km" or "800m"
    variable: str = "ppt"      # precipitation


def prism_monthly_url(cfg: PrismConfig, year: int, month: int) -> str:
    """
    Centralized PRISM web service (monthly):
    https://services.nacse.org/prism/data/get/{region}/{resolution}/{variable}/{YYYYMM}
    """
    return (
        f"https://services.nacse.org/prism/data/get/"
        f"{cfg.region}/{cfg.resolution}/{cfg.variable}/{year}{month:02d}"
    )


def download_prism_month(cfg: PrismConfig, year: int, month: int) -> Path:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.raw_dir / f"prism_{cfg.region}_{cfg.resolution}_{cfg.variable}_{year}_{month:02d}.zip"

    if out.exists():
        return out

    url = prism_monthly_url(cfg, year, month)
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, out)
    return out