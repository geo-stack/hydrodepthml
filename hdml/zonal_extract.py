# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/hydrodepthml
# =============================================================================

"""Zonal data and stats extraction capability."""

# ---- Standard imports
from pathlib import Path
from time import perf_counter

# ---- Third party imports
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from osgeo import gdal

gdal.UseExceptions()


def build_zonal_index_map(
        raster_path: Path,
        basin_gdf: gpd.GeoDataFrame,
        ) -> tuple[dict, list]:
    """
    Build an index map for basin geometries relative to a raster grid.

    For each basin, this function computes the absolute pixel
    indices (row, col) that fall inside the basin geometry. The resulting
    index map enables rapid extraction of raster values for each basin across
    multiple rasters that share the same grid and coordinate system,
    eliminating the need to re-rasterize geometries for every file (e.g.,
    when extracting daily precipitation or NDVI time series from a stack of
    co-registered GeoTIFFs).

    By default, only pixels whose centers fall within the basin geometry are
    included (all_touched=False). This is standard practice for hydrological
    zonal statistics, ensuring accurate areal weighting and preventing
    double-counting at basin boundaries.

    For very small basins that contain no pixel centers, the function
    automatically falls back to all_touched=True, which includes any pixel
    intersected by the basin boundary. This avoids empty results at the cost
    of minor edge over-sampling. Basins requiring this fallback are flagged
    in the returned 'basin_metadata'.

    Parameters
    ----------
    raster_path : Path
        Path to a representative raster (VRT/TIFF) that defines the grid.
    basin_gdf : gpd.GeoDataFrame
        GeoDataFrame containing basin geometries, indexed by basin ID.
        Must be in the same CRS as the raster.

    Returns
    -------
    zonal_index_map : dict
        Dictionary with the following structure:
        {
            'width': int,
                Raster width in pixels.
            'height': int,
                Raster height in pixels.
            'crs': str,
                Raster CRS as string (WKT or PROJ).
            'indices': dict[int, np.ndarray],
                Mapping of basin_id -> Nx2 array of [row, col] indices.
                Each array contains absolute row/col indices for pixels inside
                the basin. Basins that don't intersect the raster are omitted.
        }
    basin_metadata : dict
        Dictionary with keys:
        {
            'bad': list[int],
                Basin IDs that did not intersect the raster or had invalid
                geometry.
            'small': list[int],
                Basin IDs that required all_touched=True fallback due to
                small size relative to grid resolution.
        }
    """
    import rasterio
    from rasterio.features import geometry_window

    indices_for_geoms = {}

    # Basins that don't intersect or have invalid geometry.
    bad_basin_ids = []

    # Basins that needed all_touched=True fallback.
    small_basin_ids = []

    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        crs = src.crs

        for index, row in basin_gdf.iterrows():
            geom = row.geometry

            try:
                # Get the minimal window covering the geometry
                # (speeds up rasterize).
                win = geometry_window(src, [geom], pad_x=0, pad_y=0)
            except Exception:
                # Geometry does not intersect raster or other failure.
                bad_basin_ids.append(index)
                continue

            win_height = int(win.height)
            win_width = int(win.width)

            if win_height <= 0 or win_width <= 0:
                bad_basin_ids.append(index)
                continue

            # Rasterize the geometry into the window coordinates.
            win_transform = src.window_transform(win)

            # Try pixel-center method first
            mask = rasterize(
                [(geom, 1)],
                out_shape=(win_height, win_width),
                transform=win_transform,
                fill=0,
                all_touched=False,
                dtype='uint8'
                )

            # Fallback for small basins.
            if not mask.any():
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=(win_height, win_width),
                    transform=win_transform,
                    fill=0,
                    all_touched=True,
                    dtype='uint8'
                    )

                if not mask.any():
                    bad_basin_ids.append(index)
                    continue

                small_basin_ids.append(index)

            # Extract indices (rows/cols relative to window).
            rows, cols = np.nonzero(mask)

            # Convert to absolute row/col on the full raster.
            abs_rows = rows + int(win.row_off)
            abs_cols = cols + int(win.col_off)

            # Store as Nx2 array of [row, col] pairs.
            indices_for_geoms[int(index)] = (
                np.column_stack((abs_rows, abs_cols))
                )

    zonal_index_map = {
        'width': width,
        'height': height,
        'crs': crs.to_string(),
        'indices': indices_for_geoms
        }

    basin_metadata = {
        'bad': bad_basin_ids,
        'small': small_basin_ids
        }

    if len(small_basin_ids):
        print(f"Warning: Used all_touched=True for {len(small_basin_ids)} "
              f"small basins.")
    if len(bad_basin_ids):
        print(f"Warning: {len(bad_basin_ids)} basins excluded "
              f"(no intersection or empty after fallback).")

    return zonal_index_map, basin_metadata


def extract_zonal_means(
        raster_path: Path, zonal_index_map: dict
        ) -> np.ndarray:
    """
    Extract mean raster values for a list of geometries.

    Computes the spatial mean of raster values (e.g., NDVI, precipitation)
    within each provided geometry (e.g., watershed polygons, administrative
    boundaries). Nodata values are excluded from the mean calculation.
    Geometries that do not intersect the raster or contain only nodata will
    return NaN.

    This implementation keeps the raster file open for the entire loop,
    which is highly efficient for VRT files and large numbers of geometries.

    Parameters
    ----------
    raster_path : Path
        Path to the raster file (GeoTIFF, VRT, etc.).
    geometries : list of shapely.Geometry
        List of geometries (polygons, multipolygons) for which to extract
        raster values.  Must be in the same CRS as the raster.

    Returns
    -------
    np.ndarray
        Array of mean values, one per geometry.
    """
    n_geoms = len(zonal_index_map['indices'])
    mean_values = np.empty(n_geoms, dtype=np.float32)
    basin_ids = np.empty(n_geoms, dtype=np.int64)

    with rasterio.open(raster_path) as src:
        assert src.width == zonal_index_map['width']
        assert src.height == zonal_index_map['height']
        assert src.crs.to_string() == zonal_index_map['crs']

        data = src.read(1)
        nodata = src.nodata

        for i, basin_id in enumerate(zonal_index_map['indices'].keys()):
            basin_ids[i] = basin_id

            # Extract values.
            indices = zonal_index_map['indices'].get(basin_id)
            rows, cols = indices[:, 0], indices[:, 1]
            array = data[rows, cols]

            # Compute mean, excluding nodata
            if nodata is not None:
                valid_pixels = array[array != nodata]
            else:
                valid_pixels = array

            if valid_pixels.size > 0:
                mean_values[i] = np.mean(valid_pixels)
            else:
                # Geometry doesn't intersect raster.
                mean_values[i] = np.nan

    return mean_values, basin_ids


def extract_basin_zonal_timeseries(
        mosaic_index_path: Path,
        mosaic_dir: Path,
        basins_path: Path,
        year_start: int,
        year_end: int,
        scale_factor: float = 1.0,
        basin_id_column: str = 'HYBAS_ID'
        ) -> pd.DataFrame:
    """
    Extract basin-averaged time series from a stack of georeferenced
    raster mosaics.

    Loads basin geometries and iterates through raster files (e.g., NDVI,
    precipitation, temperature) for the specified year range, computing
    spatial mean values for each basin using zonal statistics. This function
    is optimized for processing large time series by pre-computing a zonal
    index map that is reused across all rasters.

    Parameters
    ----------
    mosaic_index_path :  Path
        Path to CSV file mapping dates (index) to raster filenames ('file'
        column). The index must be datetime-parseable.
    mosaic_dir :  Path
        Directory containing the raster mosaic files referenced in the index.
    basins_path : Path
        Path to vector file (GeoPackage, Shapefile) containing
        basin geometries.
    year_start : int
        Start year for extraction (inclusive).
    year_end : int
        End year for extraction (inclusive).
    scale_factor : float, optional
        Multiplicative scaling factor applied to raw raster
        values (e.g., 0.0001 for MODIS NDVI Int16 to physical units).
        Default is 1.0 (no scaling).
    basin_id_column : str, optional
        Name of the column in basins_path to use as basin identifiers.
        Default is 'HYBAS_ID'.

    Returns
    -------
    pd.DataFrame
        Time series DataFrame with dates as index and basin IDs as columns.
        Each cell contains the spatial mean raster value for that basin on
        that date.  Values are float32, with NaN for missing data.

    Notes
    -----
    - The zonal index map is built once using the first mosaic file, then
      reused for efficiency.
    - If mosaic grid properties (width, height, CRS) change during the time
      series (e.g., switching from regional to continental coverage), the zonal
      index map is automatically rebuilt.
    - If a mosaic file corresponds to multiple dates (e.g., 16-day composites),
      the same values are assigned to all associated dates.

    See Also
    --------
    build_zonal_index_map :  Build pixel indices for basin geometries
    extract_zonal_means :  Compute spatial means using a zonal index map
    """
    zonal_index_map = None

    mosaic_index = pd.read_csv(
        mosaic_index_path,
        index_col=0,
        parse_dates=True,
        dtype={'file': str}
        )

    years = list(range(year_start, year_end + 1))
    valid = (
        (np.isin(mosaic_index.index. year, years)) &
        (~pd.isnull(mosaic_index.file))
        )

    print('Loading basin geometries...', flush=True)
    basins_gdf = gpd.read_file(basins_path)
    basins_gdf = basins_gdf.set_index(basin_id_column, drop=True)
    basins_gdf.index = basins_gdf.index.astype(int)

    # Initialize basin time series dataframe.
    index = mosaic_index.index[valid]
    columns = list(basins_gdf.index)
    basin_timeseries = pd.DataFrame(
        data=np.full((len(index), len(columns)), np.nan, dtype='float32'),
        index=index,
        columns=columns
        )

    mosaic_fnames = np.unique(mosaic_index.file[valid])
    ntot = len(mosaic_fnames)
    for count, mosaic_fname in enumerate(mosaic_fnames):
        raster_path = mosaic_dir / mosaic_fname

        if zonal_index_map is None:
            print('Building zonal index map...', flush=True)
            zonal_index_map, _ = build_zonal_index_map(
                raster_path=raster_path, basin_gdf=basins_gdf
                )
        else:
            with rasterio.open(raster_path) as src:
                same_width = src.width == zonal_index_map['width']
                same_height = src.height == zonal_index_map['height']
                same_crs = src.crs.to_string() == zonal_index_map['crs']

            if not (same_width and same_height and same_crs):
                print('Re-building zonal index map...', flush=True)
                zonal_index_map, _ = build_zonal_index_map(
                    raster_path=raster_path, basin_gdf=basins_gdf
                    )

        t0 = perf_counter()
        dates = mosaic_index.loc[mosaic_index.file == mosaic_fname].index

        print(f"[{count+1:02d}/{ntot}] Processing {mosaic_fname}...", end=' ')

        mean_values, basin_ids = extract_zonal_means(
            mosaic_dir / mosaic_fname, zonal_index_map
            )

        # Apply scaling factor (e.g., for MODIS Int16 -> physical units)
        mean_values = mean_values * scale_factor

        assert list(basin_ids) == list(basin_timeseries.columns)
        basin_timeseries.loc[dates] = mean_values.astype('float32')

        t1 = perf_counter()
        print(f'done in {t1 - t0:0.1f} sec')

    return basin_timeseries


if __name__ == '__main__':
    import rasterio
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from hdml import __datadir__ as datadir

    # Validate 'build_zonal_index_map' function.

    vrt_index_path = datadir / 'ndvi' / 'vrt_index.csv'
    vrt_index = pd.read_csv(vrt_index_path, index_col=0, parse_dates=True)

    wtd_gdf = gpd.read_file(datadir / "data" / "wtd_obs_all.gpkg")
    wtd_gdf = wtd_gdf.set_index("ID", drop=True)

    basins_gdf = gpd.read_file(datadir / "data" / "wtd_basin_geometry.gpkg")
    basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
    basins_gdf.index = basins_gdf.index.astype(int)

    vrt_fnames = vrt_index.file
    vrt_fnames = vrt_fnames[~pd.isnull(vrt_fnames)]
    vrt_fnames = np.unique(vrt_fnames)

    vrt_path = datadir / 'ndvi' / vrt_fnames[0]

    zonal_index_map, bad_basin_ids = build_zonal_index_map(
        vrt_path, basins_gdf
        )

    # Select a couple of basin ids to test (e.g. [101, 205, 399])
    example_basin_ids = basins_gdf.index[[0, 3, 5]]
    print(example_basin_ids)

    example_basins_gdf = basins_gdf.loc[example_basin_ids]
    example_basins_gdf.to_file(
        datadir / "data" / "example_basin_geometry.gpkg",
        layer='example basin',
        driver="GPKG"
        )

    out_tif_path = vrt_path.with_suffix('.tif')
    with rasterio.open(vrt_path) as src:
        data = src.read(1)
        out_profile = src.profile.copy()
        nodata = src.nodata

        # For each basin, set all extracted pixels to nodata
        for basin_id in example_basin_ids:
            indices = zonal_index_map['indices'].get(basin_id)

            # indices: Nx2 array of [row, col]
            rows, cols = indices[:, 0], indices[:, 1]

            # Apply nodata value
            data[rows, cols] = nodata

        # Write result to a new GeoTIFF
        out_profile.update(driver='GTiff')
        with rasterio.open(out_tif_path, 'w', **out_profile) as dst:
            dst.write(data, 1)
