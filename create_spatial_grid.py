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
Generate spatial tiles for feature extraction and model training/prediction.

This script performs the following tasks:

1. Generates a regular grid of tiles covering the entire African continent
   bounding box based on the NASADEM DEM extent
2. Filters tiles to those that intersect with the African landmass geometry
   (for prediction)
3. Filters tiles to those containing water table depth observations
   (for training)

Tiles are used to partition the study area into manageable spatial units for
extracting topographic and climate features from large raster datasets. The
tile size (5000 pixels at 30m resolution = 150 km × 150 km) and overlap
(3 km) are optimized for efficient processing and seamless mosaicking.

Note: This script is OPTIONAL.

The output .gpkg files are already in the GitHub repository. This script only
needs to be run if new water table depth observations are added, which may
require generating new tiles to cover additional spatial extent.

Requirements
------------
- NASADEM DEM virtual raster (ESRI: 102022)(see 'process_dem_data.py').
- Simplified Africa landmass geometry (see 'process_usgs_coastal.py').
- Processed WTD observations (see 'process_wtd_obs.py').

Storage Requirements
--------------------
- Output GeoPackage files: minimal (~few MB total for all three files)

Outputs
-------
- 'features/tiles_africa_bbox.gpkg':
      All tiles covering the African continent bounding box
- 'features/tiles_africa_geom.gpkg':
      Tiles intersecting the African landmass (for prediction)
- 'features/tiles_wtd_obs. gpkg':
      Tiles containing water table depth observations (for training)

Note that all paths are relative to the repository's 'data/' directory
(e.g., if cloned to 'C:/Users/User/Documents/hydrodepthml/', outputs are in
'C:/Users/User/Documents/hydrodepthml/data/').

Notes
-----
- Tile size: 5000 × 5000 pixels at 30m resolution (150 km × 150 km)
- Overlap: 100 pixels (3 km) to ensure seamless feature extraction across
  tile boundaries
- Tiles are defined in ESRI:102022 (Africa Albers Equal Area Conic) projection
- The tiling grid is consistent across all datasets to enable efficient
  spatial indexing and parallel processing
"""

# ---- Third party imports
import geopandas as gpd

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.tiling import generate_tiles_bbox, filter_tiles

path_vrt_reprojected = datadir / 'dem' / 'nasadem_102022.vrt'
path_africa_geom = datadir / 'coastline' / 'africa_landmass_simple.gpkg'
path_wtd_obs = datadir / 'wtd' / 'wtd_obs_all.gpkg'

outdir = datadir / "features"
outdir.mkdir(parents=True, exist_ok=True)

# Tiles for the whole African continent bbox.
tiles_gdf_all = generate_tiles_bbox(
    input_raster=path_vrt_reprojected,
    tile_size=5000,    # in pixels
    overlap=100 * 30,  # 100 pixels at 30 meters resolution
    )
tiles_gdf_all.to_file(
    outdir / "tiles_africa_bbox.gpkg",
    driver="GPKG"
    )

# Tiles clipped to the African continent geometry.
tiles_gdf_africa = filter_tiles(
    gpd.read_file(path_africa_geom),
    tiles_gdf_all
    )
tiles_gdf_africa.to_file(
    outdir / "tiles_africa_geom.gpkg",
    driver="GPKG"
    )

# Tiles that contains water level observations.
tiles_gdf = filter_tiles(
    gpd.read_file(path_wtd_obs),
    tiles_gdf_all
    )
tiles_gdf.to_file(
    outdir / "tiles_wtd_obs.gpkg",
    driver="GPKG"
    )
