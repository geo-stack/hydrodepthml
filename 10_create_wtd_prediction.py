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
from datetime import datetime
import pickle

# ---- Third party imports
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.tiling import filter_tiles
from hdml.zonal_extract import burn_hybas_on_dem
from hdml.topo import generate_topo_features_for_tile
from hdml.wtd_helpers import recharge_period_from_basin_area
from hdml.gishelpers import raster_to_dataframe, raster_to_flat_array

# The reference date from which NDVI and precipitation averages are
# calculated.
REF_DATE = datetime(2025, 7, 31)

# Define here the lat/lon of the area for which you want to
# predict the water level.

LAT_MIN = 10.13882791720867
LAT_MAX = 11.00020461261005

LON_MIN = -13.036275477919597
LON_MAX = -12.072458460416062

# The dir where results will be saved.
PREDICT_PATH = datadir / 'predict'
PREDICT_PATH.mkdir(parents=True, exist_ok=True)


# %%

predict_bbox_gdf = gpd.GeoDataFrame(
    geometry=[box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)],
    crs='EPSG:4326'
    ).to_crs('ESRI:102022')

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

tiles_gdf = gpd.read_file(datadir / "features" / "tiles_africa_geom.gpkg")
tiles_gdf = filter_tiles(
    predict_bbox_gdf, tiles_gdf
    )

basins_path = datadir / 'basins' / 'basins_lvl12_102022.gpkg'
basins_gdf = gpd.read_file(basins_path)
basins_gdf = basins_gdf.set_index('HYBAS_ID', drop=True)
basins_gdf.index = basins_gdf.index.astype(int)

tiles_overlap_dir = datadir / 'features' / 'tiles (overlapped)'
tiles_overlap_dir.mkdir(parents=True, exist_ok=True)

tiles_cropped_dir = datadir / 'features' / 'tiles (cropped)'
tiles_cropped_dir.mkdir(parents=True, exist_ok=True)

# %%

# Generate topography-derived features for prediction tiles.
# Previously generated tiles (from training or prediction runs) are skipped.

total_tiles = len(tiles_gdf)
tile_count = 0
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

# Rasterize basin IDs onto the tile grid for fast pixel-to-basin lookup.
# Previously generated tiles (from training or prediction runs) are skipped.

name = 'hybas_id'

output_dir = tiles_cropped_dir / name
output_dir.mkdir(parents=True, exist_ok=True)

dem_dir = tiles_cropped_dir / 'dem_cond'

tile_count = 0
for _, tile_bbox_data in tiles_gdf.iterrows():
    tile_count += 1

    if total_tiles >= 100:
        progress = f"[{tile_count:03d}/{total_tiles}]"
    elif total_tiles >= 10:
        progress = f"[{tile_count:02d}/{total_tiles}]"
    else:
        progress = f"[{tile_count}/{total_tiles}]"

    tile_index = tile_bbox_data.tile_index
    ty, tx = ast.literal_eval(tile_index)
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tile_path = output_dir / tile_name
    if tile_path.exists():
        print(f"{progress} Hydro basin ID already burned for "
              f"tile {tile_index}.")
        continue

    print(f"{progress} Burning hydro basin ID for tile {tile_index}.")
    burn_hybas_on_dem(
        dem_path=dem_dir / f'dem_cond_tile_{ty:03d}_{tx:03d}.tif',
        basins_gdf=basins_gdf,
        output_path=output_dir / tile_name
        )

# %% Extract pixels to DataFrame

tile_count = 0
for _, tile_bbox_data in tiles_gdf.iterrows():
    tile_count += 1

    if total_tiles >= 100:
        progress = f"[{tile_count:03d}/{total_tiles}]"
    elif total_tiles >= 10:
        progress = f"[{tile_count:02d}/{total_tiles}]"
    else:
        progress = f"[{tile_count}/{total_tiles}]"

    print(f'{progress} Extracting data from rasters...')

    tile_index = tile_bbox_data.tile_index
    ty, tx = ast.literal_eval(tile_index)

    name = 'dem_cond'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'

    df = raster_to_dataframe(tiles_cropped_dir / name / tile_name)
    df = df.rename(columns={'value': 'point_z'})

    name = 'nearest_stream_coords'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name

    df['stream_x'] = raster_to_flat_array(tif_path, band=2)
    df['stream_y'] = raster_to_flat_array(tif_path, band=3)
    df['stream_z'] = raster_to_flat_array(tif_path, band=4)

    name = 'nearest_ridge_coords'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name

    df['ridge_x'] = raster_to_flat_array(tif_path, band=2)
    df['ridge_y'] = raster_to_flat_array(tif_path, band=3)
    df['ridge_z'] = raster_to_flat_array(tif_path, band=4)

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
        tif_path = tiles_cropped_dir / f'{name}_stats' / tile_name

        for band in bands:
            index = band_index_map[band]
            df[f'{name}_{band}'] = raster_to_flat_array(tif_path, band=index)

    name = 'hybas_id'
    tile_name = f'{name}_tile_{ty:03d}_{tx:03d}.tif'
    tif_path = tiles_cropped_dir / name / tile_name

    df['hybas_id'] = raster_to_flat_array(
        tif_path, band=1, nodata_to_nan=False)

    print(f'{progress} Calculating distances and ratios...')

    df['dist_stream'] = (
        (df.point_x - df.stream_x)**2 +
        (df.point_y - df.stream_y)**2
        )**0.5

    df['dist_top'] = (
        (df.point_x - df.ridge_x)**2 +
        (df.point_y - df.ridge_y)**2
        )**0.5

    df['ratio_dist'] = (
        df.dist_stream / (np.maximum(df.dist_top, pixel_size))
        )

    df['alt_stream'] = df.point_z - df.stream_z

    df['alt_top'] = df.ridge_z - df.point_z

    df['ratio_stream'] = (
        df['alt_stream'] / np.maximum(df['dist_stream'], pixel_size)
        )

    print(f"{progress} Adding NDVI and precipitation data "
          f"to training dataset...")

    ndvi_basin_means = pd.read_hdf(
        datadir / 'ndvi' / 'ndvi_means_africa_basins_2024-2025.h5',
        key='ndvi'
        )

    precip_basin_means = pd.read_hdf(
        datadir / 'precip' / 'precip_means_africa_basins_2024-2025.h5',
        key='precip'
        )

    hybas_ids = np.unique(df.hybas_id)

    for i, hybas_id in enumerate(hybas_ids):
        mask = (df.hybas_id == hybas_id)

        basins_data = basins_gdf.loc[hybas_id]
        basin_area_km2 = basins_data.geometry.area / 1e6

        ndays = recharge_period_from_basin_area(basin_area_km2)
        date_end = REF_DATE
        date_start = date_end - pd.Timedelta(days=ndays)
        date_range = pd.date_range(date_start, date_end)

        df.loc[mask, 'ndvi'] = np.mean(
            ndvi_basin_means.loc[date_range, hybas_id]
            )

        df.loc[mask, 'precipitation'] = np.mean(
            precip_basin_means.loc[date_range, hybas_id]
            )

    print(f'{progress} Saving predict dataframe to disk...')
    h5_path = PREDICT_PATH / f"predict_dset_tile_{ty:03d}_{tx:03d}.h5"
    df.to_hdf(h5_path, key='data', mode='w')


# %%

df = pd.read_hdf(
    "D:/Projets/hydrodepthml/data/predict/predict_dset_tile_020_004.h5",
    key='data'
    )

varlist = [
    'ratio_dist',
    'ratio_stream',
    'dist_stream',
    'alt_stream',
    'dist_top',
    'alt_top',
    'long_hessian_max',
    'long_hessian_mean',
    'long_hessian_var',
    'long_hessian_skew',
    'long_hessian_kurt',
    'long_grad_mean',
    'long_grad_var',
    'short_grad_max',
    'short_grad_var',
    'short_grad_mean',
    'stream_grad_max',
    'stream_grad_var',
    'stream_grad_mean',
    'stream_hessian_max',
    'ndvi',
    'precipitation',
    ]

X = df.loc[:, varlist].values

model_path = datadir / 'model' / 'wtd_predict_model.pkl'
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

wtd_predicted = loaded_model.predict(X)

dem_path = tiles_cropped_dir / "dem_cond" / "dem_cond_tile_020_004.tif"

output_path = datadir / 'predict' / 'pred_wtd_tile_020_004.tif'

with rasterio.open(dem_path) as dem:
    # Reshape 1D predictions to 2D grid
    height, width = dem.height, dem.width

    wtd_2d = wtd_predicted.reshape(height, width)

    # Copy metadata from DEM.
    output_meta = dem.meta.copy()

    # Update metadata for predictions.
    output_meta.update({
        'dtype': 'float32',
        'nodata': -9999,
        'compress': 'deflate'
        })

    # Write to file
    with rasterio.open(output_path, 'w', **output_meta) as dst:
        dst.write(wtd_2d. astype('float32'), 1)
