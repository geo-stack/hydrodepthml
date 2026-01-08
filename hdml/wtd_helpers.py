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
Helper utilities to read, clean, transform and plot water-table-depth (WTD)
observations. Includes functions to load per-country spreadsheets, normalize
date fields, create GeoDataFrames, filter bad points, and generate summary
plots (histograms, spatial diagnostics).
"""

# ---- Standard imports
from pathlib import Path
import os.path as osp
import re
from datetime import datetime

# ---- Third party imports
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point


COUNTRIES = ['Benin', 'Burkina', 'Guinee', 'Mali', 'Niger', 'Togo', 'Chad']
TARGET_CRS = "ESRI:102022"  # Africa Albers Equal Area Conic


def create_wtd_obs_dataset(
        datadir: Path, clip_to_geom: Path
        ) -> gpd.GeoDataFrame:
    """
    Load, assemble, and filter water table depth (WTD) observation datasets
    rom multiple countries.

    This function reads WTD observations stored as Excel files (one per
    country), combines them into a single GeoDataFrame, reprojects to a
    common CRS, removes known bad observations, and optionally clips
    the dataset to a provided spatial geometry (e.g., study area boundary).

    Parameters
    ----------
    datadir : Path
        Path to the directory containing country-level Excel files
        (e.g., 'Benin.xlsx', 'Mali.xlsx'). The directory should also contain
        a file named 'bad_obs_data.xlsx' with columns 'COUNTRY' and 'ID'
        identifying observations to exclude.
    clip_to_geom : Path, optional
        Path to a vector file (GeoPackage, Shapefile, etc.) defining the
        spatial extent to which observations should be clipped. Points falling
        outside this geometry are removed.  If None, no spatial clipping
        is applied (default:  None).

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing all valid WTD observations, with geometries
        in the CRS defined by TARGET_CRS (ESRI:102022 - Africa Albers Equal
        Area Conic). Includes all original columns from the Excel files, plus
        a 'country' column indicating the source country.

    See Also
    --------
    read_obs_wl :  Function to read country-level observation Excel files.
    """

    dfs = []
    for country in COUNTRIES:
        print(f'Loading WTD data for {country}...')
        filename = datadir / f'{country}.xlsx'
        temp = read_obs_wl(filename)
        temp['country'] = country
        dfs.append(temp)

    print('Assembling WTD data...')
    pts_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    pts_df = pts_df.reset_index(drop=True)

    original_count = len(pts_df)

    pts_gdf = gpd.GeoDataFrame(
        pts_df,
        geometry=[Point(xy) for xy in zip(pts_df.LON, pts_df.LAT)],
        crs="EPSG:4326"  # WGS84
        )
    pts_gdf = pts_gdf.to_crs(TARGET_CRS)

    # Filter bad points.
    print('Filtering bad points...')
    bad_pts_df = pd.read_excel(
        datadir / 'bad_obs_data.xlsx',
        dtype={'COUNTRY': str, 'ID': str}
        )

    countries = set(bad_pts_df.COUNTRY.values)
    ids = set(bad_pts_df.ID.values)
    mask = np.array(
        [not (row.country in countries and row.ID in ids) for
         i, row in pts_gdf.iterrows()],
        dtype=bool,
        )
    pts_gdf = pts_gdf.loc[mask]
    print(f'Removed {np.sum(~mask)} bad points.')

    # Filter points that falls outside the provided geometry.
    if clip_to_geom is not None:
        clip_to_gdf = gpd.read_file(clip_to_geom)
        if clip_to_gdf.crs != TARGET_CRS:
            clip_to_gdf = clip_to_gdf.to_crs(TARGET_CRS)

        filt_pts_gdf = gpd.clip(pts_gdf, clip_to_gdf)

        removed_count = len(pts_gdf) - len(filt_pts_gdf)
        print(f"Removed {removed_count} points that were outside "
              f"the limit of the provided geometry.")
    else:
        filt_pts_gdf = pts_gdf

    print(f"Final dataset has {len(filt_pts_gdf)} points "
          f"(from {original_count}).")

    return filt_pts_gdf


def read_obs_wl(filename) -> pd.DataFrame:
    """
    Read water level observations from an Excel file and preprocess
    the data.

    Parameters
    ----------
    filename : str
        Path to the Excel file containing water level observations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['ID', 'LON', 'LAT', 'DATE', 'NS'] and
        cleaned, parsed data.
    """
    df = pd.read_excel(
        filename,
        dtype={'DATE': str, 'NS': float, 'LON': float, 'LAT': float,
               'ID': str, 'OMES_ID': str, 'No_Ouvrage': str, 'CODE_OUVRA': str}
        )
    df.columns = ['ID', 'LON', 'LAT', 'DATE', 'NS']

    df['DATE'] = df['DATE'].apply(_normalize_date)
    df['DATE'] = pd.to_datetime(df['DATE'])

    df = df.dropna()

    return df


def _normalize_date(val) -> datetime:
    """Normalize various date string formats to a datetime object."""
    if pd.isnull(val) or val == '00:00:00' or val.strip() == '':
        return pd.NaT

    # Pattern: Year only, e.g. '1976'.
    m = re.fullmatch(r'(\d{4})', val)
    if m:
        return datetime.strptime(
            f"{m.group(1)}-01-01", "%Y-%m-%d"
            )

    # Pattern: YYYY-MM-DD HH:MM:SS or with milliseconds.
    m = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})\s+\d{2}:\d{2}:\d{2,3}', val)
    if m:
        # We keep only the date part.
        return datetime.strptime(
            f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m-%d"
            )

    # Pattern: DD/MM/YY.
    m = re.fullmatch(r'(\d{2})/(\d{2})/(\d{2})', val)
    if m:
        day, month, year = m.group(1), m.group(2), int(m.group(3))
        year = year + 2000 if year <= 25 else year + 1900

        # Convert to ISO format.
        return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")

    # Pattern: DD/MM/YYYY.
    m = re.fullmatch(r'(\d{2})/(\d{2})/(\d{4})', val)
    if m:
        # Convert to ISO format.
        return datetime.strptime(
            f"{m.group(3)}-{m.group(2)}-{m.group(1)}", "%Y-%m-%d"
            )

    raise ValueError(f'Cannot parse date with value={val}')


def plot_wl_hist(df: pd.DataFrame, country: str):
    """
    Plot a histogram of water level observation counts by decade.

    This function generates a histogram of the number of water level (WL)
    observations per decade based on the 'DATE' column of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'DATE' column of datetime type.
    country : str
        Country name to display in the title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated histogram figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    bins = list(range(1950, 2030, 10))

    counts, bins, patches = ax.hist(df['DATE'].dt.year, bins, rwidth=0.8)

    # Annotate each bar.
    for count, bin_left, patch in zip(counts, bins, patches):
        ax.text(
            patch.get_x() + (patch.get_width()/2),
            count,
            str(int(count)),
            ha='center',
            va='bottom',
        )

    # Add a little bit of space at the top of the graph for the text.
    ylim = ax.get_ylim()
    height_pixels = ax.get_window_extent().height
    data_per_pixel = (ylim[1] - ylim[0]) / height_pixels
    ypad = 10 * data_per_pixel
    ax.set_ylim(ylim[0], ylim[1] + ypad)

    ax.yaxis.grid(which='major', color='0.85')
    ax.set_axisbelow(True)
    ax.set_xlabel('Decade', labelpad=15, fontsize=12)
    ax.set_ylabel('Frequency', labelpad=15, fontsize=12)
    fig.suptitle(f'Number of WL observations for {country}')

    ax.set_xticks(bins)

    fig.tight_layout()

    return fig


def generate_wl_hist_figures(datadir: Path | str):
    countries = ['Benin', 'Burkina', 'Guinee', 'Mali', 'Niger', 'Togo']
    for country in countries:
        filename = osp.join(datadir, 'data', f'{country}.xlsx')
        df = read_obs_wl(filename)
        fig = plot_wl_hist(df, country)

        filepath = datadir / f'wl_obs_count_{country}.png'
        if not filepath.exists():
            fig.savefig(filepath, dpi=220)

    countries = ['Benin', 'Burkina', 'Guinee', 'Mali', 'Niger', 'Togo']
    dfs = []
    for country in countries:
        print(f'Loading WTD data for {country}...')
        filename = datadir / f'{country}.xlsx'
        temp = read_obs_wl(filename)
        temp['country'] = country
        dfs.append(temp)

    df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    df = df.reset_index(drop=True)

    fig = plot_wl_hist(df, 'all countries')

    filepath = datadir / 'wl_obs_count_all.png'
    if not filepath.exists():
        fig.savefig(filepath, dpi=220)
