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

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.tiling import generate_tiles_bbox, filter_tiles

outdir = datadir / "topo"
outdir.mkdir(exist_ok=True)

path_vrt_reprojected = datadir / 'dem' / 'nasadem_102022.vrt'
path_africa_geom = datadir / 'coastline' / 'africa_landmass_simple.gpkg'
path_wtd_obs = datadir / 'wtd' / 'wtd_obs_all.gpkg'

# Tiles for the whole African continent bbox.
tiles_gdf_all = generate_tiles_bbox(
    input_raster=path_vrt_reprojected,
    tile_size=5000,    # in pixels
    overlap=100 * 30,  # 100 pixels at 30 meters resolution
    )
tiles_gdf_all.to_file(
    outdir / "tiles_geom_all.gpkg",
    driver="GPKG"
    )

# Tiles clipped to the African continent geometry.
tiles_gdf_africa = filter_tiles(
    path_africa_geom,
    tiles_gdf_all
    )
tiles_gdf_africa.to_file(
    outdir / "tiles_geom_africa.gpkg",
    driver="GPKG"
    )

# Tiles that contains water level observations.
tiles_gdf = filter_tiles(
    path_wtd_obs,
    tiles_gdf_all
    )
tiles_gdf.to_file(
    outdir / "tiles_geom_training.gpkg",
    driver="GPKG"
    )
