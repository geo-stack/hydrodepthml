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
Extract and process HydroBASINS sub-basins for Africa.

This script performs the following tasks:

1. Extracts the specified basin level (default: level 12) from the
   HydroBASINS Africa dataset
2. Reprojects basins to Africa Albers Equal Area Conic (ESRI:102022)
3. Clips basins to the African continent using a two-step process:
   - First clips to Africa's bounding box for efficiency
   - Then clips to simplified African continent geometry for accuracy
4. Exports clipped basins with full HydroBASINS attributes

HydroBASINS is part of the HydroATLAS suite and provides standardized
sub-basin geometries with attributes including climate (precipitation,
temperature, evapotranspiration), hydrological (flow accumulation, stream
order), land cover, and socio-economic indicators. These attributes are used
as features for water table depth prediction.

Note: This script is OPTIONAL.

The output GeoPackage file (basins_lvl12_102022.gpkg) is already distributed
in the GitHub repository. This script only needs to be run if you want to use
a different basin level. To change the level, modify the `level` variable in
the script.


Requirements
------------
- Manual download of HydroBASINS Africa dataset (see Data Source below)
- The downloaded ZIP file must be placed in 'hdml/data/basins/'
- Simplified Africa landmass geometry (see 'process_usgs_coastal.py')


Storage Requirements
--------------------
- HydroBASINS ZIP archive (Africa): ~536 MB
- Extracted shapefiles: temporary (deleted after processing)
- Output basin GeoPackage (level 12): ~349 MB


Data Source
-----------
HydroBASINS Version 1c (Africa region, levels 1-12)
Download: https://www.hydrosheds.org/products/hydrobasins
Documentation: https://www.hydrosheds.org/products/hydroatlas

HydroBASINS is a global, standardized sub-basin boundary dataset derived from
HydroSHEDS data.  It provides hierarchical basin geometries at 12 nested levels
with associated HydroATLAS attributes for water resources management, modeling,
and environmental assessment.

Basin levels range from 1 (largest basins) to 12 (smallest sub-basins).
Level 12 provides the highest spatial resolution for basin-scale analysis.

Required file: 'hybas_af_lev01-12_v1c.zip'


Outputs
-------
- 'basins/basins_lvl12_102022.gpkg':
      African sub-basins at level 12 with full HydroBASINS attributes
      (ESRI:102022 projection)

Note that all paths are relative to the repository's 'data/' directory
(e.g., if cloned to 'C:/Users/User/Documents/hydrodepthml/', outputs are in
'C:/Users/User/Documents/hydrodepthml/data/').
"""

# ---- Standard imports
import zipfile
import shutil

# ---- Third party imports
import geopandas as gpd
from shapely.geometry import box

# ---- Local imports
from hdml import __datadir__ as datadir

BASINS_PATH = datadir / 'basins'
BASINS_PATH.mkdir(parents=True, exist_ok=True)


# Extract the .zip archive.

zip_path = BASINS_PATH / 'hybas_af_lev01-12_v1c.zip'
zip_fname = zip_path.name
zip_url = 'https://www.hydrosheds.org/products/hydrobasins'

if not zip_path.exists():
    raise FileNotFoundError(
        f"\n[HydroBASINS Database Missing]\n"
        f"\nCould not locate required ZIP archive:\n"
        f"    {zip_path}\n"
        f"\nTo resolve:\n"
        f"  1. Download the file '{zip_fname}' from:\n"
        f"     {zip_url}\n"
        f"  2. Move it to the folder:\n"
        f"     {BASINS_PATH}\n"
        )

# Extract basins level 12 from the BasinATLAS.
extract_dir = BASINS_PATH / zip_path.stem
if not extract_dir.exists():
    print("Extrating zip archive...", flush=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


# %%

level = 12
layer_name = f'hybas_af_lev{level:02d}_v1c'

# Clip the basins to the African continent.

africa_gdf = gpd.read_file(
    datadir / 'coastline' / 'africa_landmass_simple.gpkg'
    )

print(f'Reading {layer_name} from {extract_dir.name}...', flush=True)
basins_gdf = gpd.read_file(extract_dir, layer=layer_name)
print('Number of basins:', len(basins_gdf), flush=True)

print('Projecting to ESRI:102022...', flush=True)
basins_gdf = basins_gdf.to_crs('ESRI:102022')

# Clipping to the non-simplifed African continent shape is way too long,
# so we need to do this in two steps and use a simplified shape.

print("Clipping to African continent bbox...", flush=True)
basins_gdf_bbox = gpd.clip(basins_gdf, box(*africa_gdf.total_bounds))
print('Number of basins:', len(basins_gdf_bbox), flush=True)

print("Clipping to simplified African continent shape...", flush=True)
basins_africa = gpd.clip(basins_gdf_bbox, africa_gdf.union_all())
print('Number of basins:', len(basins_africa), flush=True)

print("Saving results to file...", flush=True)
basins_path = BASINS_PATH / f'basins_lvl{level:02d}_102022.gpkg'
basins_africa.to_file(basins_path, layer=layer_name, driver="GPKG")

# Clean up temp files.
shutil.rmtree(extract_dir)
