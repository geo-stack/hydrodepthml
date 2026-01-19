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
Assemble and process water table depth (WTD) observations for model training.

This script performs the following tasks:

1. Loads water table depth observations from multiple West African countries
   (Excel files)
2. Combines observations into a single GeoDataFrame and removes known bad
   observations
3. Clips observations to the African continent boundary
4. Spatially joins observations with HydroATLAS level 12 sub-basins to extract
   basin attributes (climate, area, etc.)
5. Calculates the recharge period and climatic data time window needed for
   each observation based on basin area
6. Filters observations to retain only those with valid date ranges
   (2003–2024) for feature extraction
7. Exports processed observations and corresponding basin geometries

The resulting dataset is used for training the water table depth prediction
model, with basin-level climate attributes and time-specific precipitation
and NDVI features.

Note: This script is OPTIONAL.

The output .gpkg files and input Excel data files are already distributed in
the GitHub repository. This script only needs to be run if new water table
depth observations are added.  If adding data for a new country, the file path
must be added to the INPUTFILES dictionary.


Requirements
------------
- Country-level WTD observation Excel files must be present in 'hdml/data/wtd/'
- Bad observations list ('bad_obs_data.xlsx') must be available
- Africa landmass geometry (see 'process_usgs_coastal. py')
- HydroATLAS level 12 basins (see 'process_hydro_basins.py')


Storage Requirements
--------------------
- Input Excel files: minimal (a few MB per country)
- Output GeoPackage files: minimal (~few MB)


Data Source
-----------
Water table depth observations were collected from groundwater monitoring
networks across seven West African countries:  Togo, Benin, Burkina Faso,
Chad, Guinea, Mali, and Niger. Each country's dataset is stored as an
Excel file containing point locations (lat/lon), measurement dates, and
observed water table depths.

Basin-level climate attributes are derived from HydroATLAS Version 1.0:
- pre_mm_syr: Annual average precipitation (mm)
- tmp_dc_syr: Annual average air temperature (°C)
- pet_mm_syr: Annual average potential evapotranspiration (mm)
- aet_mm_syr: Annual average actual evapotranspiration (mm)
- ari_ix_sav:  Global aridity index
- cmi_ix_syr: Annual average climate moisture index

See:  https://www.hydrosheds.org/products/hydroatlas


Outputs
-------
- 'wtd/wtd_obs_all.gpkg':
      Processed water table depth observations with basin attributes and
      climatic data time windows
- 'wtd/wtd_basin_geometry.gpkg':
      HydroATLAS level 12 basin geometries for basins containing observations

Note that all paths are relative to the repository's 'data/' directory
(e.g., if cloned to 'C:/Users/User/Documents/hydrodepthml/', outputs are in
'C:/Users/User/Documents/hydrodepthml/data/').


Notes
-----
- Observations before 2003 or after 2024 are excluded to ensure sufficient
  historical climate data (MODIS starts in 2000) and model relevance
- The recharge period (days of climate data needed before observation date)
  is calculated based on basin area using an empirical relationship
- Each observation is assigned to a single HydroATLAS sub-basin using
  spatial join ('within' predicate)
"""

# ---- Standard imports

# ---- Third party imports
import numpy as np
import pandas as pd
import geopandas as gpd

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.wtd_helpers import (
    create_wtd_obs_dataset, recharge_period_from_basin_area)

INPUTFILES = {
    'Togo': datadir / 'wtd' / 'Togo.xlsx',
    'Benin': datadir / 'wtd' / 'Benin.xlsx',
    'Burkina': datadir / 'wtd' / 'Burkina.xlsx',
    'Chad': datadir / 'wtd' / 'Chad.xlsx',
    'Guinee': datadir / 'wtd' / 'Guinee.xlsx',
    'Mali': datadir / 'wtd' / 'Mali.xlsx',
    'Niger': datadir / 'wtd' / 'Niger.xlsx'
    }

gwl_gdf = create_wtd_obs_dataset(
    input_filepaths=INPUTFILES,
    bad_obs_path=datadir / 'wtd' / 'bad_obs_data.xlsx',
    clip_to_geom=datadir / 'coastline' / 'africa_landmass.gpkg'
    )

# Join information about sub-basin level 12 from the HydroATLAS database.
basins_path = datadir / 'basins' / 'basins_lvl12_102022.gpkg'
basins_gdf = gpd.read_file(basins_path)
basins_gdf['basin_area_km2'] = basins_gdf.geometry.area / 1e6

joined = gpd.sjoin(gwl_gdf, basins_gdf, how='left', predicate='within')

# Remove columns that we do not want from the HydroATLAS database.

# syr: annual average in sub-basin
# tmp_dc_syr: air temperature annual average in sub-basin
# pre_mm_syr: precipitation annual average in sub-basin
# pet_mm_syr: potential evapotranspiration annual average in sub-basin
# aet_mm_syr: actual evapotranspiration annual average in sub-basin
# ari_ix_sav: global aridity index
# cmi_ix_syr: climate moisture index annual average in sub-basin

columns = list(gwl_gdf.columns)
columns += ['pre_mm_syr', 'tmp_dc_syr', 'pet_mm_syr', 'aet_mm_syr',
            'ari_ix_sav', 'cmi_ix_syr', 'basin_area_km2', 'HYBAS_ID']

gwl_gdf = joined[columns].copy()
gwl_gdf['HYBAS_ID'] = gwl_gdf['HYBAS_ID'].astype(int)

# Calculate the period for which we will need daily climatic data
# to compute the 'ndvi' and 'precipitation' features.
for index, row in gwl_gdf.iterrows():
    ndays = recharge_period_from_basin_area(row.basin_area_km2)
    date_start = row.DATE - pd.Timedelta(days=ndays)
    gwl_gdf.loc[index, 'climdata_period_days'] = ndays
    gwl_gdf.loc[index, 'climdata_date_start'] = date_start


# We filter out measurements that are before 2003 and after 2025.
year_min = 2002
year_max = 2025
mask = ((gwl_gdf.climdata_date_start.dt.year > year_min) &
        (gwl_gdf.DATE.dt.year < year_max))
gwl_gdf = gwl_gdf[mask]

original_count = len(mask)
removed_count = np.sum(mask)
print(f"Removed {removed_count} points (from {original_count}) that "
      f"were before {year_min + 1} or after {year_max - 1}.")
print(f'Final dataset has {len(gwl_gdf)} points.')

# Save the water table observations dataset.
gwl_gdf.to_file(datadir / "wtd" / "wtd_obs_all.gpkg", driver="GPKG")

# Save the basins geometry (we keep only the ones with water level obs).
basins_gdf = basins_gdf.set_index('HYBAS_ID', drop=True)
basins_gdf = basins_gdf.loc[gwl_gdf['HYBAS_ID'].unique()]
basins_gdf = basins_gdf['geometry']

basins_gdf.to_file(datadir / "wtd" / "wtd_basin_geometry.gpkg", driver="GPKG")
