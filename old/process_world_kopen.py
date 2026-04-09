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

# ---- Standard imports
import subprocess
import shutil
import zipfile

# ---- Third party imports
import numpy as np
from osgeo import gdal
import rasterio
from scipy.ndimage import distance_transform_edt

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.gishelpers import create_pyramid_overview


# Requirements
# ------------
# - Manual download of 'hess-11-1633-2007-supplement' (see Data Source below)
# - 7-Zip executable (7za.exe) - included in the repository
# - The downloaded ZIP file must be placed in 'hdml/data/coastline/'

# Data Source
# https://hess.copernicus.org/articles/11/1633/2007/

gdal.UseExceptions()

DEM_PATH = datadir / 'dem' / 'nasadem_102022.vrt'


CLIMZ_DIR = datadir / 'climate_zones'
CLIMZ_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = CLIMZ_DIR / 'world_koppen.tiff'


# %% Extract 'hess-11-1633-2007-supplement' content

print("Extract USGS global islands database...")

# Extract with 7zip (because zipfile does not support the 'mpk' format)
exepath = datadir / '7za.exe'

# Extract the .zip archive.
zip_fname = 'hess-11-1633-2007-supplement.zip'
zip_path = CLIMZ_DIR / zip_fname
zip_url = 'https://hess.copernicus.org/articles/11/1633/2007/'

if not zip_path.exists():
    raise FileNotFoundError(
        f"\n[Updated world map of the Köppen-Geiger climate"
        f" classification Missing]\n"
        f"\nCould not locate required ZIP archive:\n"
        f"    {zip_path}\n"
        f"\nTo resolve:\n"
        f"  1. Download the file '{zip_fname}' from:\n"
        f"     {zip_url}\n"
        f"  2. Move it to the folder:\n"
        f"     {CLIMZ_DIR}\n"
        )


extract_dir = CLIMZ_DIR / 'hess-11-1633-2007-supplement'
if not extract_dir.exists():
    print("Extrating zip archive...", flush=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


# %%

# Assign CRS to the Köppen raster (we assume that it is WGS84)

print('Creating a GeoTiff file and set CRS to WGS84...')

adf_folder_path = extract_dir / 'Raster files' / 'world_koppen'
temp_with_crs = CLIMZ_DIR / 'koppen_with_crs.tif'

translate_options = gdal.TranslateOptions(
    format='GTiff',
    outputSRS='EPSG:4326',
    creationOptions=['COMPRESS=LZW', 'TILED=YES']
    )
ds = gdal.Translate(
    str(temp_with_crs),
    str(adf_folder_path),
    options=translate_options
    )
ds.FlushCache()
del ds

# Delete extract folder since we don't need it anymore.
shutil.rmtree(extract_dir)

# %%

print("Warp Köppen raster to match DEM grid...")

temp_warped = CLIMZ_DIR / 'koppen_warped_temp.tif'

# Read DEM to get target grid parameters.
with rasterio.open(DEM_PATH) as dem:
    target_crs = dem.crs
    target_transform = dem.transform
    target_width = dem.width
    target_height = dem.height
    target_bounds = dem.bounds
    dem_nodata = dem.nodata

warp_options = gdal.WarpOptions(
    format='GTiff',
    dstSRS=str(target_crs),
    xRes=target_transform.a,
    yRes=abs(target_transform.e),
    outputBounds=(target_bounds.left, target_bounds.bottom,
                  target_bounds.right, target_bounds.top),
    width=target_width,
    height=target_height,
    resampleAlg='near',  # Use 'near' for categorical data like climate zones
    outputType=gdal.GDT_Byte,  # Use Byte (0-255) for climate zones
    creationOptions=['COMPRESS=LZW', 'TILED=YES']
    )

ds = gdal.Warp(str(temp_warped), str(temp_with_crs), options=warp_options)
ds.FlushCache()
del ds

# %%

print("Fill gaps using nearest neighbor where DEM has data...")

with rasterio.open(DEM_PATH) as dem:
    dem_data = dem.read(1)
    dem_mask = dem_data != dem_nodata
    dem_nodata_mask = dem_data == dem_nodata

with rasterio.open(temp_warped) as src:
    koppen_data = src.read(1)
    profile = src.profile

koppen_nodata_mask = (koppen_data == 255)
needs_filling = koppen_nodata_mask & dem_mask

if np.any(needs_filling):
    # Find nearest valid Köppen value for each nodata pixel.
    valid_mask = ~koppen_nodata_mask
    indices = distance_transform_edt(
        ~valid_mask, return_distances=False, return_indices=True)

    koppen_filled = koppen_data[tuple(indices)]

    koppen_data = np.where(needs_filling, koppen_filled, koppen_data)

# Set Köppen value to nodata where dem is nodata:
koppen_data[dem_nodata_mask] = 255

with rasterio.open(OUTPUT_PATH, 'w', **profile) as dst:
    dst.write(koppen_data, 1)

# Clean up temporary files.
temp_with_crs.unlink()
temp_warped.unlink()

# %%

print("Creating a pyramid overview...")
create_pyramid_overview('D:/Projets/hydrodepthml/data/climate_zones/koppen_warped_temp.tif')
