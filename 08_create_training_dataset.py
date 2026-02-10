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

"""
Generate topographic and climate features for water table depth model training.

This script performs the following tasks:

1. Generates topographic features from NASADEM DEM for all tiles containing
   water table depth observations (streams, ridges, terrain statistics)
2. Extracts topographic feature values at each observation point location
3. Calculates derived features: distances to streams/ridges, elevation
   differences, and topographic ratios
4. Adds basin-averaged NDVI and precipitation time series for each observation
5. Exports the complete training dataset with all features

The resulting dataset contains all predictor variables (topographic, climate,
and basin attributes) needed to train the water table depth prediction model.

Requirements
------------
- NASADEM DEM virtual raster (ESRI:102022)(see 'process_nasadem.py')
- Processed WTD observations (see 'process_wtd_obs.py')
- Tile index for WTD observations (see 'generate_tiles.py')
- NDVI basin means (training) ('process_ndvi_data.py')
- Precipitation basin means (training) (see 'process_precip_data.py')

Storage Requirements
--------------------
- Final training dataset: minimal (~few MB)
- Trained model (~20 MB)

Feature Extraction
------------------
Topographic features derived from NASADEM:
- point_z, stream_z, ridge_z: Elevations (m) at point, nearest stream, and
  nearest ridge
- stream_x, stream_y, ridge_x, ridge_y: Coordinates of nearest stream/ridge
- dist_stream, dist_top: Euclidean distances (m) to stream and ridge
- ratio_dist: Ratio of distance to stream vs. distance to ridge
- alt_stream, alt_top: Elevation differences (m) from point to stream/ridge
- ratio_stream: Slope ratio (elevation difference / distance to stream)
- Terrain statistics: Hessian and gradient statistics over multiple scales
  (long-range, short-range, stream-focused)

Climate features (basin-averaged over recharge period):
- ndvi: Mean NDVI from MODIS MOD13Q1
- precipitation: Mean daily precipitation (mm) from CHIRPS

Outputs
-------
- 'features/tiles (overlapped)/<feature_name>/':
      Topographic feature tiles with overlap for seamless extraction
- 'features/tiles (cropped)/<feature_name>/':
      Topographic feature tiles cropped to tile boundaries
- 'model/wtd_obs_training_dataset.csv':
      Final training dataset with all features (topographic + climate)

Note that all paths are relative to the repository's 'data/' directory
(e.g., if cloned to 'C:/Users/User/Documents/hydrodepthml/', outputs are in
'C:/Users/User/Documents/hydrodepthml/data/').
"""

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

with rasterio.open(nasadem_mosaic_path) as src:
    # The horizontal and vertical resolution should be the same.
    pixel_size = src.res[0]


gwl_gdf = gpd.read_file(datadir / "wtd" / "wtd_obs_all.gpkg")

tiles_gdf = gpd.read_file(datadir / "features" / "tiles_wtd_obs.gpkg")

tiles_overlap_dir = datadir / 'features' / 'tiles (overlapped)'
tiles_overlap_dir.mkdir(parents=True, exist_ok=True)

tiles_cropped_dir = datadir / 'features' / 'tiles (cropped)'
tiles_cropped_dir.mkdir(parents=True, exist_ok=True)

model_dir = datadir / 'model'
model_dir.mkdir(parents=True, exist_ok=True)

filter_sigma = 1
stream_treshold = 500

OUTPUT_FILE = (
    model_dir /
    f"wtd_obs_training_dataset_"
    f"sig{filter_sigma}_"
    f"st{stream_treshold}.csv"
    )

# %%

# Generate the topo-derived features for all tiles containing at least
# one valid WTD observation.

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
        extract_streams_treshold=stream_treshold,
        gaussian_filter_sigma=filter_sigma,
        ridge_size=30,
        )


# %%

# Extract topo-derived features from the pre-processed tiles.

gwl_gdf['point_x'] = gwl_gdf.geometry.x
gwl_gdf['point_y'] = gwl_gdf.geometry.y

joined = gpd.sjoin(
    gwl_gdf, tiles_gdf[['tile_index', 'geometry']],
    how='left', predicate='within'
    )
joined = joined.drop(columns=['index_right'])

ntot = len(np.unique(joined.tile_index))
count = 1
for tile_idx, group in joined.groupby('tile_index'):
    print(f"[{count}/{ntot}] Processing tile index: {tile_idx}...")

    coords = [(geom.x, geom.y) for geom in group.geometry]
    ty, tx = ast.literal_eval(tile_idx)

    name = 'dem'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'elev'] = values[:, 0]

    name = 'world_koppen'
    tile_name = f'{name}.tiff'
    tif_path = datadir / 'climate_zones' / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)), dtype=int)
        values[values == src.nodata] = -1

        gwl_gdf.loc[group.index, name] = values[:, 0]

    name = 'dem_cond'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'point_z'] = values[:, 0]

    name = 'nearest_stream_coords'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'stream_x'] = values[:, 2]
        gwl_gdf.loc[group.index, 'stream_y'] = values[:, 3]
        gwl_gdf.loc[group.index, 'stream_z'] = values[:, 4]

    name = 'nearest_ridge_coords'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'ridge_x'] = values[:, 2]
        gwl_gdf.loc[group.index, 'ridge_y'] = values[:, 3]
        gwl_gdf.loc[group.index, 'ridge_z'] = values[:, 4]

    name = 'nearest_divide_coords'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'divide_x'] = values[:, 2]
        gwl_gdf.loc[group.index, 'divide_y'] = values[:, 3]
        gwl_gdf.loc[group.index, 'divide_z'] = values[:, 4]

    name = 'wetness_index'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name
    with rasterio.open(tif_path) as src:
        values = np.array(list(src.sample(coords)))
        values[values == src.nodata] = np.nan

        gwl_gdf.loc[group.index, 'wetness_index'] = values[:, 0]

    band_index_map = {
        'min': 0,
        'max': 1,
        'mean': 2,
        'var': 3,
        'skew': 4,
        'kurt': 5
        }

    name_bands = {
        'long_dem': ['max', 'min', 'mean', 'var', 'skew', 'kurt'],
        'short_dem': ['max', 'min', 'mean', 'var', 'skew', 'kurt'],
        'stream_dem': ['max', 'min', 'mean', 'var', 'skew', 'kurt'],
        'long_hessian': ['max', 'mean', 'var', 'skew', 'kurt'],
        'long_grad': ['mean', 'var'],
        'short_grad': ['max', 'var', 'mean'],
        'stream_grad': ['max', 'var', 'mean'],
        'stream_hessian': ['max']
        }

    for name, bands in name_bands.items():
        tile_name = f'{name}_stats_tile_{ty:03d}_{tx:03d}.tif'
        tif_path = tiles_cropped_dir / f'{name}_stats' / tile_name

        with rasterio.open(tif_path) as src:
            values = np.array(list(src.sample(coords)))
            values[values == src.nodata] = np.nan

        for band in bands:
            index = band_index_map[band]
            gwl_gdf.loc[group.index, f'{name}_{band}'] = values[:, index]

    gwl_gdf.loc[group.index, 'tile_index'] = f'{ty:03d}_{tx:03d}'

    count += 1


# Calculate distances and ratios.

print('Calculate distances and ratios...')

gwl_gdf['dist_stream'] = (
    (gwl_gdf.point_x - gwl_gdf.stream_x)**2 +
    (gwl_gdf.point_y - gwl_gdf.stream_y)**2
    )**0.5

gwl_gdf['dist_top'] = (
    (gwl_gdf.point_x - gwl_gdf.ridge_x)**2 +
    (gwl_gdf.point_y - gwl_gdf.ridge_y)**2
    )**0.5

gwl_gdf['dist_divide'] = (
    (gwl_gdf.point_x - gwl_gdf.divide_x)**2 +
    (gwl_gdf.point_y - gwl_gdf.divide_y)**2
    )**0.5

gwl_gdf['ratio_dist'] = (
    gwl_gdf.dist_stream / (np.maximum(gwl_gdf.dist_top, pixel_size))
    )

gwl_gdf['alt_stream'] = gwl_gdf.point_z - gwl_gdf.stream_z

gwl_gdf['alt_top'] = gwl_gdf.ridge_z - gwl_gdf.point_z

gwl_gdf['alt_divide'] = gwl_gdf.divide_z - gwl_gdf.point_z

gwl_gdf['ratio_stream'] = (
    gwl_gdf['alt_stream'] / np.maximum(gwl_gdf['dist_stream'], pixel_size)
    )

gwl_gdf['ratio_stream_divide'] = (
    gwl_gdf.dist_stream / (gwl_gdf.dist_divide + gwl_gdf.dist_stream)
    )


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
    basin_id = int(row.HYBAS_ID)

    # Add mean daily NDVI values (at the basin scale).
    ndvi_values = ndvi_means_wtd_basins.loc[date_range, basin_id]
    gwl_gdf.loc[index, 'ndvi'] = np.mean(ndvi_values)

    # Add mean daily PRECIP values (at the basin scale).
    precip_values = precip_means_wtd_basins.loc[date_range, basin_id]
    gwl_gdf.loc[index, 'precipitation'] = np.mean(precip_values)

    # Add yearly (2 years prior) average NDVI and precip average
    # values (at the basin scale).
    date_end = row.DATE
    date_start = date_end - pd.Timedelta(days=360 * 2)
    date_range = pd.date_range(date_start, date_end)

    ndvi_values = ndvi_means_wtd_basins.loc[date_range, basin_id]
    gwl_gdf.loc[index, 'ndvi_yrly_avg'] = np.nanmean(ndvi_values)

    precip_values = precip_means_wtd_basins.loc[date_range, basin_id]
    gwl_gdf.loc[index, 'precip_yrly_avg'] = np.nanmean(precip_values)


# %%
print("Saving dataset to file...")
gwl_gdf.to_csv(OUTPUT_FILE, index=False)
