from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import urllib.request
import urllib.error


@dataclass(frozen=True)
class PrismConfig:
    raw_dir: Path = Path("data/raw/prism")
    region: str = "us"
    resolution: str = "4km"
    variable: str = "ppt"
    timeout_s: int = 120           # per request timeout
    retries: int = 5               # number of retries
    backoff_s: float = 3.0         # base backoff seconds


def prism_monthly_url(cfg: PrismConfig, year: int, month: int) -> str:
    return (
        f"https://services.nacse.org/prism/data/get/"
        f"{cfg.region}/{cfg.resolution}/{cfg.variable}/{year}{month:02d}"
    )


def download_prism_month(cfg: PrismConfig, year: int, month: int) -> Path:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.raw_dir / f"prism_{cfg.region}_{cfg.resolution}_{cfg.variable}_{year}_{month:02d}.zip"

    # If already downloaded and looks non-empty, reuse it
    if out.exists() and out.stat().st_size > 50_000:
        return out

    url = prism_monthly_url(cfg, year, month)

    last_err: Exception | None = None
    for attempt in range(1, cfg.retries + 1):
        try:
            print(f"Downloading: {url} (attempt {attempt}/{cfg.retries})")
            urllib.request.urlretrieve(url, out)  # uses global socket timeout below
            # Basic sanity: PRISM zips are not tiny; avoid caching HTML error pages
            if out.exists() and out.stat().st_size > 50_000:
                return out
            else:
                raise RuntimeError(f"Downloaded file too small (likely error page): {out} ({out.stat().st_size} bytes)")
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last_err = e
            # Remove partial/bad file before retrying
            if out.exists():
                try:
                    out.unlink()
                except Exception:
                    pass
            sleep_s = cfg.backoff_s * attempt
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download PRISM {year}-{month:02d} after {cfg.retries} retries. Last error: {last_err}")