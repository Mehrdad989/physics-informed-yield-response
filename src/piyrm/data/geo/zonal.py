from __future__ import annotations

import pandas as pd
import geopandas as gpd
import rasterio
from exactextract import exact_extract


def zonal_mean(
    *,
    polygons: gpd.GeoDataFrame,
    raster_path: str,
    id_col: str,
) -> pd.DataFrame:
    """
    Compute mean of raster within each polygon using exactextract.

    Works with exactextract versions that return a GeoJSON-like table:
    columns ['type', 'properties'], where 'properties' is a dict containing
    the requested stats + any include_cols.
    """
    with rasterio.open(raster_path) as src:
        if polygons.crs is None:
            raise ValueError("polygons must have a CRS set")

        polys = polygons.to_crs(src.crs) if polygons.crs != src.crs else polygons

        stats = exact_extract(
            src,
            polys,
            ["mean"],
            include_cols=[id_col],
        )

    df = pd.DataFrame(stats)

    # Newer/alternate exactextract output: a 'properties' dict per row
    if "properties" in df.columns:
        props = pd.json_normalize(df["properties"])
        df2 = props.copy()
    else:
        # Older output: stats already flat
        df2 = df.copy()

    # Find id column
    if id_col not in df2.columns:
        # try case-insensitive match
        candidates = [c for c in df2.columns if c.lower() == id_col.lower()]
        if not candidates:
            raise RuntimeError(
                f"ID column '{id_col}' not found after normalization. "
                f"Got columns: {list(df2.columns)}"
            )
        id_found = candidates[0]
    else:
        id_found = id_col

    # Find mean column
    if "mean" in df2.columns:
        mean_col = "mean"
    else:
        mean_candidates = [c for c in df2.columns if "mean" in c.lower()]
        if not mean_candidates:
            raise RuntimeError(
                f"No mean-like column found after normalization. "
                f"Got columns: {list(df2.columns)}"
            )
        mean_col = mean_candidates[0]

    out = df2[[id_found, mean_col]].rename(columns={id_found: id_col, mean_col: "raster_mean"})
    return out