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

# ---- Standard imports
import ast

# ---- Third party imports
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.topo import generate_topo_features_for_tile

nasadem_mosaic_path = datadir / 'dem' / 'nasadem_102022.vrt'
if not nasadem_mosaic_path.exists():
    raise FileNotFoundError(
        "Make sure to run 'process_dem_data.py' before running this "
        "script to download and process DEM data."
        )

gwl_gdf = gpd.read_file(datadir / "wtd" / "wtd_obs_all.gpkg")

tiles_gdf = gpd.read_file(datadir / "topo" / "tiles_topo_wtd_obs.gpkg")

with rasterio.open(nasadem_mosaic_path) as src:
    # The horizontal and vertical resolution should be the same.
    pixel_size = src.res[0]

# %%

# Generate the topo-derived features for all tiles containing at least
# one valid WTD observation.

tiles_overlap_dir = datadir / 'topo' / 'tiles (cropped)'
tiles_overlap_dir.mkdir(parents=True, exist_ok=True)

tiles_cropped_dir = datadir / 'topo' / 'tiles (overlapped)'
tiles_cropped_dir.mkdir(parents=True, exist_ok=True)

tile_count = 0
total_tiles = len(tiles_gdf)
for _, tile_bbox_data in tiles_gdf.iterrows():
    tile_count += 1

    if total_tiles >= 100:
        progress = f"[{tile_count:03d}/{total_tiles}]"
    elif total_tiles >= 10:
        progress = f"[{tile_count:02d}/{total_tiles}]"
    else:
        progress = f"[{tile_count}/{total_tiles}]"

    generate_topo_features_for_tile(
        tile_bbox_data=tile_bbox_data,
        dem_path=nasadem_mosaic_path,
        crop_tile_dir=tiles_cropped_dir,
        ovlp_tile_dir=tiles_overlap_dir,
        print_affix=progress,
        extract_streams_treshold=500,
        gaussian_filter_sigma=1,
        ridge_size=30,
        )

# %%

# Extract top-derived features from the pre-processed tiles.

gwl_gdf['point_x'] = gwl_gdf.geometry.x
gwl_gdf['point_y'] = gwl_gdf.geometry.y

joined = gpd.sjoin(
    gwl_gdf, tiles_gdf[['tile_index', 'geometry']],
    how='left', predicate='within'
    )
joined = joined.drop(columns=['index_right'])

input_dir = datadir / 'topo' / 'tiles (cropped)'

ntot = len(np.unique(joined.tile_index))
count = 1
for tile_idx, group in joined.groupby('tile_index'):
    print(f"[{count}/{ntot}] Processing tile index: {tile_idx}...")

    coords = [(geom.x, geom.y) for geom in group.geometry]
    ty, tx = ast.literal_eval(tile_idx)

    name = 'dem_cond'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = input_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        pixel_size = src.res
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'point_z'] = values[:, 0]

    name = 'nearest_stream_coords'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = input_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'stream_x'] = values[:, 2]
        gwl_gdf.loc[group.index, 'stream_y'] = values[:, 3]
        gwl_gdf.loc[group.index, 'stream_z'] = values[:, 4]

    name = 'nearest_ridge_coords'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = input_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'ridge_x'] = values[:, 2]
        gwl_gdf.loc[group.index, 'ridge_y'] = values[:, 3]
        gwl_gdf.loc[group.index, 'ridge_z'] = values[:, 4]

    band_index_map = {
        'min': 0,
        'max': 1,
        'mean': 2,
        'var': 3,
        'skew': 4,
        'kurt': 5
        }

    name_bands = {
        'long_hessian': ['max', 'mean', 'var', 'skew', 'kurt'],
        'long_grad': ['mean', 'var'],
        'short_grad': ['max', 'var', 'mean'],
        'stream_grad': ['max', 'var', 'mean'],
        'stream_hessian': ['max']
        }

    for name, bands in name_bands.items():
        tile_name = f'{name}_stats_tile_{ty:03d}_{tx:03d}.tif'
        tif_path = input_dir / f'{name}_stats' / tile_name

        with rasterio.open(tif_path) as src:
            values = np.array(list(src.sample(coords)))
            values[values == src.nodata] = np.nan

        for band in bands:
            index = band_index_map[band]
            gwl_gdf.loc[group.index, f'{name}_{band}'] = values[:, index]

    count += 1


# %%

# Calculate distances and ratios.

gwl_gdf['dist_stream'] = (
    (gwl_gdf.point_x - gwl_gdf.stream_x)**2 +
    (gwl_gdf.point_y - gwl_gdf.stream_y)**2
    )**0.5

gwl_gdf['dist_top'] = (
    (gwl_gdf.point_x - gwl_gdf.ridge_x)**2 +
    (gwl_gdf.point_y - gwl_gdf.ridge_y)**2
    )**0.5

gwl_gdf['ratio_dist'] = (
    gwl_gdf.dist_stream / (np.maximum(gwl_gdf.dist_top, pixel_size))
    )

gwl_gdf['alt_stream'] = gwl_gdf.point_z - gwl_gdf.stream_z

gwl_gdf['alt_top'] = gwl_gdf.ridge_z - gwl_gdf.point_z

gwl_gdf['ratio_stream'] = (
    gwl_gdf['alt_stream'] / np.maximum(gwl_gdf['dist_stream'], pixel_size)
    )

gwl_gdf.to_file(datadir / "wtd_obs_training_dataset.gpkg", driver="GPKG")
gwl_gdf.to_csv(datadir / "wtd_obs_training_dataset.csv")


# %%

# Add precip and ndvi avg sub-basin values for each water level observation.

print("Adding NDVI and precipitation data to training dataset...")


ndvi_means_wtd_basins = pd.read_hdf(
    datadir / 'ndvi' / 'ndvi_means_wtd_basins_2000-2025.h5',
    key='ndvi'
    )

precip_means_wtd_basins = pd.read_hdf(
    datadir / 'precip' / 'precip_means_wtd_basins_2000-2025.h5',
    key='precip'
    )

for index, row in gwl_gdf.iterrows():
    date_range = pd.date_range(row.climdata_date_start, row.DATE)
    basin_id = str(int(row.HYBAS_ID))

    # Add mean daily NDVI values (at the basin scale).
    ndvi_values = ndvi_means_wtd_basins.loc[date_range, basin_id]
    gwl_gdf.loc[index, 'ndvi'] = np.mean(ndvi_values)

    # Add mean daily PRECIP values (at the basin scale).
    precip_values = precip_means_wtd_basins.loc[date_range, basin_id]
    gwl_gdf.loc[index, 'precipitation'] = np.mean(precip_values)

gwl_gdf.to_csv(datadir / "wtd_obs_training_dataset.csv")
