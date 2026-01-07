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

# You need to download the 'USGSEsriWCMC_GlobalIslands_v3_mpk.zip'
# archive from:
# https://www.sciencebase.gov/catalog/item/63bdf25dd34e92aad3cda273
# and copy it to the folder './hdml/data/coastline'

# See also:
# https://data.usgs.gov/datacatalog/data/USGS:63bdf25dd34e92aad3cda273
# https://pubs.usgs.gov/publication/70202401


# ---- Standard imports.
import shutil
import subprocess

# ---- Third party imports.
import geopandas as gpd
import pandas as pd

# ---- Local imports.
from hdml import __datadir__ as datadir

COAST_DIR = datadir / 'coastline'
COAST_DIR.mkdir(parents=True, exist_ok=True)

# %% Extract global islands database

print("Extract USGS global islands database...")

# Extract with 7zip (because zipfile does not support the 'mpk' format)
exepath = datadir / '7za.exe'

# Extract the .zip archive.
zip_fname = 'USGSEsriWCMC_GlobalIslands_v3_mpk.zip'
zip_path = COAST_DIR / zip_fname
zip_url = 'https://www.sciencebase.gov/catalog/item/63bdf25dd34e92aad3cda273'

if not zip_path.exists():
    raise FileNotFoundError(
        f"\n[USGS Global Islands Database Missing]\n"
        f"\nCould not locate required ZIP archive:\n"
        f"    {zip_path}\n"
        f"\nTo resolve:\n"
        f"  1. Download the file '{zip_fname}' from:\n"
        f"     {zip_url}\n"
        f"  2. Move it to the folder:\n"
        f"     {COAST_DIR}\n"
        )

command = f'"{exepath}" x "{zip_path}" -o"{COAST_DIR}"'
result = subprocess.run(
    command, capture_output=True, text=True, shell=True, check=True
    )

# Extract the .mpk archive.
mpk_path = COAST_DIR / 'USGSEsriWCMC_GlobalIslands_v3.mpk'
extract_dir = COAST_DIR / 'USGSEsriWCMC_GlobalIslands_v3'
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir(parents=True, exist_ok=True)

command = f'"{exepath}" x "{mpk_path}" -o"{extract_dir}"'
result = subprocess.run(
    command, capture_output=True, text=True, shell=True, check=True
    )

mpk_path.unlink()

# %% Extract African continent from global dataset

print('Extract African continent from global dataset...')

gdb_path = extract_dir / 'v108/globalislandsfix.gdb'

ADD_BIG_ISLANDS = False

# Fetch the shapefile for the African continent.
gdf_africa = gpd.read_file(
    gdb_path, layer='USGSEsriWCMC_GlobalIslandsv2_Continents'
    )
gdf_africa = gdf_africa.loc[gdf_africa.OBJECTID == 2]

# Add big islands.
if ADD_BIG_ISLANDS:
    africa_bbox = gdf_africa.total_bounds
    gdf_big_isl = gpd.read_file(
        gdb_path, layer='USGSEsriWCMC_GlobalIslandsv2_BigIslands')
    gdf_big_isl = gdf_big_isl.cx[
        africa_bbox[0] - 10**6:africa_bbox[2] + 10**6,
        africa_bbox[1] - 10**6:africa_bbox[3] + 10**6
        ]

    gdf_africa = gpd.GeoDataFrame(
        pd.concat([gdf_africa, gdf_big_isl], ignore_index=True),
        crs=gdf_africa.crs
        )

# Reproject and save.
gdf_africa = gdf_africa.to_crs('ESRI:102022')  # Africa Albers Equal Area Conic
gdf_africa.to_file(COAST_DIR / 'africa_landmass.gpkg', driver='GPKG')

shutil.rmtree(extract_dir)

# %% Simplify geometry

print('Creating a simplified geometry of the African continent...')

# Create a simplified geometry of the African continent to speed up
# spatial operations. The original, highly detailed geometry makes clipping
# subbasins to Africa slow (in 'process_hydro_basins.py'). Buffering and
# simplifying reduces complexity, making subsequent processing much faster.

africa_simple_path = datadir / 'coastline' / 'africa_landmass_simple.gpkg'

# Simplify the geometry by buffering outward and inward (to remove small
# geometry artifacts), then apply a topology-ignoring simplification.

gdf_africa = gpd.read_file(COAST_DIR / 'africa_landmass.gpkg')
gdf_africa_simple = gdf_africa.buffer(5000)
gdf_africa_simple = gdf_africa_simple.buffer(-5000)
gdf_africa_simple = gdf_africa_simple.simplify(1000, preserve_topology=False)

gdf_africa_simple.to_file(africa_simple_path)
