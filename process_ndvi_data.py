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
Download, process, and extract MODIS NDVI (MOD13Q1) data for Africa.

This script performs the following tasks:

1. Downloads MODIS MOD13Q1 (250m, 16-day composite) HDF files
   from NASA EarthData
2. Extracts NDVI bands and converts them to GeoTIFF format
3. Mosaics tiles by date and reprojects to Africa Albers Equal Area
   Conic (ESRI:102022)
4. Extracts basin-averaged NDVI time series for HydroATLAS level 12 sub-basins

Two datasets are produced:

- Prediction dataset ('ndvi_means_africa_basins_2024-2025.h5'):
    Daily NDVI averages for all African basins, covering 2024–2025.
- Training dataset ('ndvi_means_wtd_basins_2000-2025.h5'):
    Daily NDVI averages for basins containing water table observations,
    covering 2000–2025.

Note: This script is OPTIONAL.

The output .h5 files are already distributed in the GitHub repository as Git
Large File Storage (LFS) files and do not need to be regenerated unless you
want to add more water level observations for model training or update/change
the reference date (2025-07-31) for water table depth prediction. If adding
new observation sites, you must also update the MODIS_TILE_NAMES variable to
include the corresponding tiles and year ranges (see inline comments and
https://modis-land.gsfc.nasa.gov/MODLAND_grid.html).


Requirements
------------
- NASA EarthData account (free): https://urs.earthdata.nasa.gov/
- MODIS tile coverage is defined in MODIS_TILE_NAMES based on observation
  locations and prediction domain
- HydroATLAS level 12 basins must be available (see 'process_hydro_basins.py')

To use this script, you must have a valid NASA Earthdata account. You will be
prompted to provide your Earthdata username and password for authentication.


Storage Requirements
--------------------
- MODIS MOD13Q1 HDF files (250m): ~1.09 TB
- Extracted NDVI TIF files (250m): ~148 GB
- NDVI mosaic TIF files (250m): ~154 GB
- Compiled NDVI means (HDF5): ~709 MB (647 MB + 62 MB)
- Total peak storage: ~1.39 TB

Note: MODIS HDF and extracted TIF files can be deleted or archived after
mosaics are produced to recover ~1.24 TB of disk space.

Data Source
-----------
MODIS MOD13Q1 Version 6.1 (Terra satellite, 250m resolution, 16-day composite)
Documentation: https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-006
Coverage: 2000–present (Terra launch:  late 1999).


Outputs
-------
- 'ndvi_tiles_index.csv':
      Index mapping dates to individual MODIS tile GeoTIFFs
- 'ndvi_mosaic_index.csv':
      Index mapping daily dates to reprojected mosaics
- 'ndvi_means_wtd_basins.h5':
      Basin-averaged NDVI for training (2000–2025)
- 'ndvi_means_africa_basins.h5':
      Basin-averaged NDVI for prediction (2024–2025)

Note that all paths are relative to the repository's 'data/' directory
(e.g., if cloned to 'C:/Users/User/Documents/hydrodepthml/', outputs are in
'C:/Users/User/Documents/hydrodepthml/data/').
"""

# ---- Standard imports
from pathlib import Path
import time
from time import perf_counter

# ---- Third party imports
from osgeo import gdal
import pandas as pd
import rasterio

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.ed_helpers import (
    earthaccess_login, MOD13Q1_hdf_to_geotiff, get_mod13q1_hdf_urls)
from hdml.zonal_extract import extract_basin_zonal_timeseries

# MODIS_TILE_NAMES specifies which MODIS tiles to download NDVI data for,
# along with the year ranges:
# - For 2024 and 2025: Download all tiles covering Africa for static
#   water depth prediction.
# - For tiles where water level observations are available, download data
#   for 2000–2025 for model training.
# If more observation sites are added to the training dataset, update this
# list to include the corresponding tiles and years.
# See https://modis-land.gsfc.nasa.gov/MODLAND_grid.html

predict_year_range = (2024, 2025)
training_year_range = (2000, 2025)

MODIS_TILE_NAMES = [
    # row 05
    ('h17v05', *predict_year_range),
    ('h18v05', *predict_year_range),
    ('h19v05', *predict_year_range),
    ('h20v05', *predict_year_range),
    # row 06
    ('h16v06', *predict_year_range),
    ('h17v06', *predict_year_range),
    ('h18v06', *predict_year_range),
    ('h19v06', *predict_year_range),
    ('h20v06', *predict_year_range),
    ('h21v06', *predict_year_range),
    # row 07
    ('h16v07', *training_year_range),
    ('h17v07', *training_year_range),
    ('h18v07', *training_year_range),
    ('h19v07', *training_year_range),
    ('h20v07', *training_year_range),
    ('h21v07', *predict_year_range),
    ('h22v07', *predict_year_range),
    # row 08
    ('h16v08', *training_year_range),
    ('h17v08', *training_year_range),
    ('h18v08', *training_year_range),
    ('h19v08', *training_year_range),
    ('h20v08', *predict_year_range),
    ('h21v08', *predict_year_range),
    ('h22v08', *predict_year_range),
    # row 09
    ('h18v09', *predict_year_range),
    ('h19v09', *predict_year_range),
    ('h20v09', *predict_year_range),
    ('h21v09', *predict_year_range),
    ('h22v09', *predict_year_range),
    # row 10
    ('h19v10', *predict_year_range),
    ('h20v10', *predict_year_range),
    ('h21v10', *predict_year_range),
    ('h22v10', *predict_year_range),
    # row 11
    ('h19v11', *predict_year_range),
    ('h20v11', *predict_year_range),
    ('h21v11', *predict_year_range),
    # row 12
    ('h19v12', *predict_year_range),
    ('h20v12', *predict_year_range)
    ]

NDVI_DIR = datadir / 'ndvi'
NDVI_DIR.mkdir(parents=True, exist_ok=True)

HDF_DIR = Path("F:/BanqueMondiale (HydroDepthML)/MODIS MOD13Q1 HDF 250m")

TIF_DIR = Path("F:/BanqueMondiale (HydroDepthML)/MODIS NDVI TIF 250m")

VRT_DIR = NDVI_DIR / 'vrt'
VRT_DIR.mkdir(parents=True, exist_ok=True)

MOSAIC_DIR = NDVI_DIR / 'mosaic'
MOSAIC_DIR.mkdir(parents=True, exist_ok=True)

tif_file_index_path = NDVI_DIR / 'ndvi_tiles_index.csv'
mosaic_index_path = NDVI_DIR / 'ndvi_mosaic_index.csv'


# %%

# Authenticate to Earthdata and get available datasets

print("Authenticating with NASA Earthdata...")
earthaccess = earthaccess_login()

# Get the list of available hDF names from the NDVI MODIS dataset for the
# entire African continent.

hdf_urls = {}
ntot = len(MODIS_TILE_NAMES)
i = 0
print()
for tile_name, year_from, year_to in MODIS_TILE_NAMES:
    progress = f"[{i+1:02d}/{ntot}]"

    print(f"{progress} Getting HDF urls for tile {tile_name}...")
    count = 0
    while True:
        try:
            tile_hdf_urls = get_mod13q1_hdf_urls(tile_name, year_from, year_to)
        except RuntimeError as err:
            count += 1
            if count > 3:
                print(f"{progress}  Failed after {count} attempts.")
                raise err
            wait_time = 2 ** count  # Exponential backoff:  2, 4, 8 seconds
            print(f"{progress} RuntimeError: The CMR query failed "
                  f"(try #{count}). Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            break

    hdf_urls.update(tile_hdf_urls)
    print(f"{progress} Found {len(tile_hdf_urls)} granules for {tile_name}")

    i += 1


# %%

# Download the NDVI MODIS tiles and convert to GeoTIFF (skip if they exist).

tif_file_index = pd.DataFrame(
    index=pd.MultiIndex.from_tuples([], names=['date_start', 'date_end'])
    )

i = 0
n = len(hdf_urls)
for hdf_name, url in hdf_urls.items():
    progress = f"[{i+1:02d}/{n}]"

    tif_fpath = TIF_DIR / (hdf_name + '.tif')
    if tif_fpath.exists():
        with rasterio.open(tif_fpath) as src:
            print(f'{progress} NDVI data already downloaded and processed.')
            meta_dict = src.tags()
            tile_name = meta_dict.get('tile_name')
            date_start = meta_dict.get('date_start')
            date_end = meta_dict.get('date_end')
        tif_file_index.loc[(date_start, date_end), tile_name] = tif_fpath.name
        i += 1
        continue

    # Download the MODIS HDF file.
    hdf_fpath = HDF_DIR / (hdf_name + '.hdf')
    if not hdf_fpath.exists():
        print(f'{progress} Downloading MODIS HDF file...')
        try:
            earthaccess.download(url, HDF_DIR, show_progress=False)
        except Exception:
            print(f'{progress} Failed to download NDVI data for {hdf_name}.')
            break

    # Convert MODIS HDF file to GeoTIFF.
    print(f'{progress} Converting to GeoTIFF...')
    tile_name, date_start, date_end = MOD13Q1_hdf_to_geotiff(
        hdf_fpath, 0, tif_fpath)

    tif_file_index.loc[(date_start, date_end), tile_name] = tif_fpath.name
    i += 1


tif_file_index = tif_file_index.sort_index()
tif_file_index.to_csv(tif_file_index_path)


# %%

# Generate the tiled GeoTIFF mosaic.

tif_file_index = pd.read_csv(tif_file_index_path, index_col=[0, 1])

mosaic_index = pd.DataFrame(
    columns=['file', 'ntiles'],
    index=pd.date_range('2000-01-01', '2025-12-31')
    )

ntot = len(tif_file_index)
i = 0
for index, row in tif_file_index.iterrows():
    start = index[0].replace('-', '')
    end = index[1].replace('-', '')

    # Define the list of tiles to add to the mosaic.
    tif_paths = [
        TIF_DIR / tif_fname for tif_fname in row.values if
        not pd.isnull(tif_fname)
        ]

    # Define the name of the final mosaic and check if it exists.
    mosaic_path = MOSAIC_DIR / f"NDVI_MOD13Q1_{start}_{end}_ESRI102022.tif"

    # Get only dates that exist in mosaic_index.
    date_range = pd.date_range(*index)
    valid_dates = mosaic_index.index.intersection(date_range)

    if mosaic_path.exists():
        print(f"[{i+1:02d}/{ntot}] Mosaic already exists for {index[0]}.")
        mosaic_index.loc[valid_dates, 'file'] = mosaic_path.name
        mosaic_index.loc[valid_dates, 'ntiles'] = len(tif_paths)
        i += 1
        continue

    t0 = perf_counter()
    print(f"[{i+1:02d}/{ntot}] Producing a mosaic for {index[0]}...", end=' ')

    # Define the name of the VRT file.
    vrt_path = VRT_DIR / f"NDVI_MOD13Q1_{start}_{end}.vrt"

    # Build a VRT first.
    if not vrt_path.exists():
        ds = gdal.BuildVRT(vrt_path, tif_paths)
        ds.FlushCache()
        del ds

    # Reprojected and assemble the tiles into a mosaic.
    warp_options = gdal.WarpOptions(
        dstSRS='ESRI:102022',  # Africa Albers Equal Area Conic
        format='GTiff',
        resampleAlg='bilinear',
        creationOptions=[
            'COMPRESS=DEFLATE',
            'TILED=YES',
            'BIGTIFF=YES'
            ]
        )

    ds_reproj = gdal.Warp(
        str(mosaic_path),
        str(vrt_path),
        options=warp_options
        )
    ds_reproj.FlushCache()
    del ds_reproj

    # Update the VRT file index.
    date_range = pd.date_range(*index)
    valid_dates = mosaic_index.index.intersection(date_range)
    mosaic_index.loc[valid_dates, 'file'] = mosaic_path.name
    mosaic_index.loc[valid_dates, 'ntiles'] = len(tif_paths)

    i += 1
    t1 = perf_counter()
    print(f'done in {t1 - t0:0.1f} sec')

mosaic_index = mosaic_index.dropna(how='all')
mosaic_index.to_csv(mosaic_index_path)


# %%

# Calculate the daily mean NDVI for all the basins of the African continent
# for the 2024 and 2025 years.
basins_path = datadir / 'basins' / 'basins_lvl12_102022.gpkg'
year_start, year_end = predict_year_range
ndvi_means_africa_basins = extract_basin_zonal_timeseries(
    mosaic_index_path=mosaic_index_path,
    mosaic_dir=MOSAIC_DIR,
    basins_path=basins_path,
    year_start=year_start,
    year_end=year_end,
    scale_factor=0.0001,   # MODIS Int16 scale to physical NDVI
    basin_id_column='HYBAS_ID'
    )

fname = f'ndvi_means_africa_basins_{year_start}-{year_end}.h5'
ndvi_means_africa_basins.to_hdf(NDVI_DIR / fname, key='ndvi', mode='w')


# %%

# Calculate the daily mean NDVI for the basins where water level observations
# are available for 2000–2025.
basins_path = datadir / 'wtd' / 'wtd_basin_geometry.gpkg'
year_start, year_end = training_year_range
ndvi_means_wtd_basins = extract_basin_zonal_timeseries(
    mosaic_index_path=mosaic_index_path,
    mosaic_dir=MOSAIC_DIR,
    basins_path=basins_path,
    year_start=year_start,
    year_end=year_end,
    scale_factor=1,
    basin_id_column='HYBAS_ID'
    )

fname = f'ndvi_means_wtd_basins_{year_start}-{year_end}.h5'
ndvi_means_wtd_basins.to_hdf(NDVI_DIR / fname, key='ndvi', mode='w')
