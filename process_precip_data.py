# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel
# =============================================================================

"""
Download, process, and extract CHIRPS daily precipitation data for Africa.

This script performs the following tasks:

1. Downloads CHIRPS v3.0 daily precipitation GeoTIFF files from the
   Climate Hazards Center.
2. Clips to Africa's bounding box and reprojects to Africa Albers Equal Area
   Conic (ESRI:102022)
3. Indexes files by date for efficient time series extraction
4. Extracts basin-averaged precipitation time series for HydroATLAS
   level 12 sub-basins

Two datasets are produced:

- Training dataset ('precip_means_wtd_basins_2000-2025.h5'):
      Daily precipitation averages for basins containing water table
      observations, covering 2000–2025.
- Prediction dataset ('precip_means_africa_basins_2024-2025.h5'):
      Daily precipitation averages for all African basins, covering 2024–2025

Note: This script is OPTIONAL.

The output .h5 files are already distributed in the GitHub repository as Git
Large File Storage (LFS) files and do not need to be regenerated unless you
want to add more water level observations for model training or update/change
the reference date (2025-07-31) for water table depth prediction.


Requirements
------------
- HydroATLAS level 12 basins must be available (see 'process_hydro_basins.py')
- Africa landmass geometry for bounding box extraction
  (see 'process_usgs_coastal.py')


Storage Requirements
--------------------
- CHIRPS GeoTIFF files (Africa, 2000–2025): ~32.6 GB
- Compiled precipitation means (HDF5): ~682 MB (620 MB + 62 MB)
- Total storage: ~33.3 GB


Data Source
-----------
CHIRPS v3.0 (Climate Hazards Group InfraRed Precipitation with Station data)
Resolution: ~0.05° (~5.5 km at equator)
Temporal coverage: 1981–present (daily)
Documentation: https://www.chc.ucsb.edu/data/chirps
Data combines satellite imagery with in-situ station data for improved
accuracy data-scarce regions.


Outputs
-------
- 'precip_mosaic_index.csv':
      Index mapping dates to clipped/reprojected precipitation GeoTIFFs
- 'precip_means_wtd_basins_2000-2025.h5':
      Basin-averaged precipitation for training
- 'precip_means_africa_basins_2024-2025.h5':
      Basin-averaged precipitation for prediction


Notes
-----
- Files are downloaded only once and skipped if they already exist
- Precipitation values are in millimeters (mm) and require no scaling
- Global files are clipped to Africa's bounding box to reduce storage
  requirements
- The script uses zonal statistics to compute spatial averages within
  each basin
- Update YEAR_RANGE if extending the time series beyond 2000–2025
"""

# ---- Standard imports.
from datetime import datetime

# ---- Third party imports.
import pandas as pd
import requests
from bs4 import BeautifulSoup
import geopandas as gpd
import numpy as np

# ---- Local imports.
from hdml import __datadir__ as datadir
from hdml.gishelpers import clip_and_project_raster
from hdml.zonal_extract import extract_basin_zonal_timeseries

# See https://www.chc.ucsb.edu/data/chirps.

PRECIP_DIR = datadir / 'precip'
PRECIP_DIR.mkdir(parents=True, exist_ok=True)

basins_gdf = gpd.read_file(datadir / "wtd" / "wtd_basin_geometry.gpkg")
basins_gdf = basins_gdf.set_index("HYBAS_ID", drop=True)
basins_gdf.index = basins_gdf.index.astype(int)

mosaic_index_path = PRECIP_DIR / 'precip_mosaic_index.csv'

# Read Africa landmass and get bounding box.
africa_gdf = gpd.read_file(datadir / 'coastline' / 'africa_landmass.gpkg')
africa_shape = africa_gdf.union_all()
africa_bbox = africa_shape.bounds  # (minx, miny, maxx, maxy)

base_url = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat"

YEAR_RANGE = list(range(2000, 2026))


# %%

# Download the CHIRPS daily sat data for year in YEAR_RANGE.

mosaic_index = pd.DataFrame(
    columns=['file'],
    index=pd.date_range('2000-01-01', '2025-12-31')
    )
mosaic_index.index.name = 'date'

# Fetch available CHIRPS files.
print('Fetching available files on the CHIRPS server...')
chirp_files = {}
for year in YEAR_RANGE:
    year_url = base_url + f'/{year}'

    # Get the list of tif files available for download.
    resp = requests.get(year_url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    files = [a['href'] for a in
             soup.find_all("a") if
             a['href'].endswith(".tif")]

    chirp_files[year] = files

# Download and process the files.
for year, files in chirp_files.items():

    print(f"Downloading CHIRPS daily precip tifs for year {year}...")

    for file in files:
        file_url = base_url + f'/{year}/{file}'

        dtime = datetime.strptime(file_url[-14:-4], '%Y.%m.%d')

        global_tif_fpath = PRECIP_DIR / file

        tif_fpath = PRECIP_DIR / f'{year}' / file
        tif_fpath.parent.mkdir(parents=True, exist_ok=True)

        mosaic_index.loc[dtime, 'file'] = f'{year}/{file}'

        # Skip if already downloaded.
        if tif_fpath.exists():
            print(f"[{str(dtime)[:10]}] Precip data already "
                  f"downloaded and processed....")
            continue

        print(f"[{str(dtime)[:10]}] Downloading tif file...", end='')

        try:
            resp = requests.get(file_url, stream=False, timeout=60)
            resp.raise_for_status()
            with open(global_tif_fpath, "wb") as fp:
                fp.write(resp.content)
        except Exception as err:
            mosaic_index.loc[dtime, 'file'] = np.nan
            mosaic_index.to_csv(mosaic_index_path)
            raise err

        clip_and_project_raster(
            global_tif_fpath,
            tif_fpath,
            output_crs='ESRI:102022',
            clipping_bbox=africa_bbox
            )

        global_tif_fpath.unlink()

        print(' done')

print('All precip tif file downloaded successfully.')

print('Saving mosaic index to file...', end='')
mosaic_index.to_csv(mosaic_index_path)
print('done')

# %%

# Calculate the daily mean precipitation for all the basins of the African
# continent for the 2024 and 2025 years.
basins_path = datadir / 'basins' / 'basins_lvl12_102022.gpkg'
if not basins_path.exists():
    raise FileNotFoundError(
        "Make sure to run 'process_hydro_basins.py' to generate the "
        "the 'basins_lvl12_102022.gpkg' file."
        )

year_start = 2024
year_end = 2025

precip_means_africa_basins = extract_basin_zonal_timeseries(
    mosaic_index_path=mosaic_index_path,
    mosaic_dir=PRECIP_DIR,
    basins_path=basins_path,
    year_start=year_start,
    year_end=year_end,
    scale_factor=1,
    basin_id_column='HYBAS_ID'
    )

fname = f'precip_means_africa_basins_{year_start}-{year_end}.h5'
precip_means_africa_basins.to_hdf(PRECIP_DIR / fname, key='precip', mode='w')

# %%

# Calculate the daily mean precipitation for the basins where water level
# observations are available for 2000–2025.
basins_path = datadir / 'wtd' / 'wtd_basin_geometry.gpkg'
if not basins_path.exists():
    raise FileNotFoundError(
        "Make sure to run 'process_wtd_observations.py' to generate the "
        "the 'wtd_basin_geometry.gpkg' file."
        )

year_start = 2000
year_end = 2025

precip_means_wtd_basins = extract_basin_zonal_timeseries(
    mosaic_index_path=mosaic_index_path,
    mosaic_dir=PRECIP_DIR,
    basins_path=basins_path,
    year_start=year_start,
    year_end=year_end,
    scale_factor=1,
    basin_id_column='HYBAS_ID'
    )

fname = f'precip_means_wtd_basins_{year_start}-{year_end}.h5'
precip_means_wtd_basins.to_hdf(PRECIP_DIR / fname, key='precip', mode='w')
