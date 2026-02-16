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
Extract and process Köppen-Geiger climate zones for the African continent.

This script performs the following tasks:

1. Extracts the Köppen-Geiger climate zone raster from
   the "hess-11-1633-2007-supplement" archive.
2. Converts the raster to GeoTIFF format and assigns the WGS84 CRS.
3. Projects the raster to Africa Albers Equal Area Conic (ESRI:102022) and
   clips to the African bounding box.
4. Fills all NoData cells in the climate raster with the nearest valid class
   using distance-based nearest neighbor.
5. Clips the filled raster to the precise African continent shape, including
   all pixels that touch the landmass polygon.
6. Outputs a clean, analysis-ready raster of Köppen climate zones spatially
   aligned with other project datasets.

The output raster is specifically prepared to ensure that every land pixel in
the 30 m NASADEM of Africa has a corresponding valid Köppen climate zone,
enabling straightforward pixel-to-pixel association.

Note: This script is OPTIONAL.
The complete source "hess-11-1633-2007-supplement.zip" and preprocessed
output "koppen_climate_zones.tif" are included in the repository. Only
rerun this script if you wish to regenerate the climate raster, use a
different mask geometry, or test new reprojection/fill settings.

Requirements
------------
- Manual download of 'hess-11-1633-2007-supplement.zip' (see Data Source)
- The downloaded ZIP archive must be placed in 'hdml/data/climate_zones/'
- Simplified Africa landmass geometry (see 'process_usgs_coastal.py')

Storage Requirements
--------------------
- 'hess-11-1633-2007-supplement.zip': ~6.8 MB
- Intermediate rasters: minimal, deleted after processing
- Output GeoTIFF: 'climate_zones/koppen_climate_zones.tif' ~21 KB

Data Source
-----------
World Map of the Köppen-Geiger Climate Classification (Beck et al. 2007)
Download: https://hess.copernicus.org/articles/11/1633/2007/
Required file: 'hess-11-1633-2007-supplement.zip'

The Köppen-Geiger classification provides a globally standardized climate
zone raster widely used in geoscientific research, hydrology, and modeling.

Outputs
-------
- 'climate_zones/koppen_climate_zones.tif':
      Raster of Köppen-Geiger climate classes clipped and filled for the
      entire African continent (projected to ESRI:102022, with all pixels
      covering Africa assigned a valid climate class)

Note: All paths are relative to the repository's 'data/' directory
(e.g., if cloned to 'C:/Users/User/Documents/hydrodepthml/', outputs are in
'C:/Users/User/Documents/hydrodepthml/data/').
"""

# ---- Standard imports
import shutil
import zipfile

# ---- Third party imports
import geopandas as gpd
import numpy as np
from osgeo import gdal
import rasterio
from rasterio.mask import mask
from scipy.ndimage import distance_transform_edt

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.gishelpers import clip_and_project_raster

gdal.UseExceptions()

CLIMZ_DIR = datadir / 'climate_zones'
CLIMZ_DIR.mkdir(parents=True, exist_ok=True)


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

# Convert to GeoTiff and assign CRS to the Köppen raster (we assume that
# it is WGS84)

print('Creating a GeoTiff file and set CRS to WGS84...')

adf_folder_path = extract_dir / 'Raster files' / 'world_koppen'
with_crs_path = CLIMZ_DIR / 'koppen_with_crs.tif'

translate_options = gdal.TranslateOptions(
    format='GTiff',
    outputSRS='EPSG:4326',
    creationOptions=['COMPRESS=LZW', 'TILED=YES']
    )
ds = gdal.Translate(
    str(with_crs_path),
    str(adf_folder_path),
    options=translate_options
    )
ds.FlushCache()
del ds

# Delete extract folder since we don't need it anymore.
shutil.rmtree(extract_dir)

# %%

print("Clipping to the African continent bounding box...")

# Read Africa landmass and get bounding box.
africa_gdf = gpd.read_file(
    datadir / 'coastline' / 'africa_landmass_simple.gpkg')
africa_shape = africa_gdf.union_all()
africa_bbox = africa_shape.bounds  # (minx, miny, maxx, maxy)

africa_102022_path = CLIMZ_DIR / 'koppen_africa_albers.tif'

clip_and_project_raster(
    input_raster=with_crs_path,
    output_raster=africa_102022_path,
    output_crs='ESRI:102022',
    clipping_bbox=africa_bbox,
    resample_alg='nearest'
    )

# Remove temp files.
with_crs_path.unlink(missing_ok=True)
with_crs_path.with_suffix('.tif.aux.xml').unlink(missing_ok=True)


# %%

print("Filling all nodata cells...")

with rasterio.open(africa_102022_path) as src:
    arr = src.read(1)
    nodata = src.nodata

    # Create a mask of nodata pixels.
    nodata_mask = arr == nodata
    if np.any(nodata_mask):
        # Find indices of nearest non-nodata for each pixel
        distance, (inds_y, inds_x) = distance_transform_edt(
            nodata_mask, return_indices=True)
        arr_filled = arr[inds_y, inds_x]
    else:
        arr_filled = arr.copy()

    # Prepare metadata for output.
    out_meta = src.meta.copy()
    out_meta.update({'compress': 'lzw', 'tiled': True})

# Save the filled raster
filled_path = CLIMZ_DIR / 'koppen_africa_albers_filled.tif'
with rasterio.open(filled_path, 'w', **out_meta) as dest:
    dest.write(arr_filled, 1)

# Remove temp files.
africa_102022_path.unlink(missing_ok=True)
africa_102022_path.with_suffix('.tif.aux.xml').unlink(missing_ok=True)


# %%

print('Clipping to the African continent shape...')

final_path = CLIMZ_DIR / 'koppen_climate_zones.tif'

with rasterio.open(filled_path) as src:
    # Ensure matching CRS
    africa_gdf = africa_gdf.to_crs(src.crs)

    # Get geometry as GeoJSON-like dicts
    africa_geom = [
        feature["geometry"] for feature in
        africa_gdf.__geo_interface__["features"]
        ]

    # Mask with all_touched=True
    out_image, out_transform = mask(
        src,
        africa_geom,
        crop=True,
        nodata=src.nodata,
        all_touched=True,
        )

    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "compress": "lzw",
        "tiled": True,
        })

# Write clipped raster
with rasterio.open(final_path, "w", **out_meta) as dest:
    dest.write(out_image)


# Remove temp files.
filled_path.unlink(missing_ok=True)
filled_path.with_suffix('.tif.aux.xml').unlink(missing_ok=True)
