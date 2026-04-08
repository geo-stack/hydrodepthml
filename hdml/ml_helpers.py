# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel_water_table_ml
# =============================================================================

"""
ml_helpers.py

Helper utilities for Machine Learning training and evaluation.
"""

# ---- Standard imports
import os
from pathlib import Path

# ---- Third party imports
from matplotlib.transforms import ScaledTranslation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Local imports.


def plot_ns_distribution(depths):

    std = np.std(depths)
    mean = np.mean(depths)

    # low_cutoff = mean - std
    # high_cutoff = mean + std

    # Percentile-based boundaries
    low_cutoff = np.percentile(depths, 5)   # 10th percentile
    high_cutoff = np.percentile(depths, 70)  # 90th percentile

    classes = np.empty(len(depths), dtype=object)

    classes[depths <= low_cutoff] = 'shallow'
    classes[depths >= high_cutoff] = 'deep'
    classes[(depths > low_cutoff) & (depths < high_cutoff)] = 'middle'

    counts_classes = [f'shallow\n<{low_cutoff: 0.1f} m',
                      f'middle\n] {low_cutoff:0.1f} - {high_cutoff: 0.1f} ] m',
                      f'deep\n>{high_cutoff: 0.1f} m']
    counts_values = [
        np.sum(classes == 'shallow'),
        np.sum(classes == 'deep'),
        np.sum(classes == 'middle')
        ]

    n = len(counts_values)

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(list(range(n)), counts_values, color='skyblue')
    ax.set_xlabel('Classes', fontsize=12, labelpad=10)
    ax.set_ylabel('Nombre', fontsize=12, labelpad=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks(list(range(n)))
    ax.set_xticklabels(counts_classes)

    # Ajout padding vertical (10 % au-dessus du max).
    ax.set_ylim(top=np.max(counts_values) * 1.11)

    # Ajout de la valeur et proportion au dessus de chaque barre.
    ntot = len(depths)
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        count = bar.get_height()
        perc = count / ntot * 100
        ax.text(x, count,
                f"{count:,}".replace(",", " ") + "\n" + f"({perc:0.1f}%)",
                ha='center', va='bottom', fontsize=10,
                transform=ax.transData + ScaledTranslation(
                    0, 1/72, ax.figure.dpi_scale_trans)
                )

    fig.tight_layout()






def plot_feature_importance(importances: np.ndarray, features: list):
    """
    Plot the feature importances as a horizontal bar chart.

    This function visualizes the relative importance of input features,
    typically as computed by a machine learning model (e.g., random forest,
    gradient boosting). Each feature's importance is shown as a horizontal bar,
    sorted from most to least important.

    Parameters
    ----------
    importances : np.ndarray
        Array of feature importance scores (length = number of features).
    features : list of str
        List of feature names (strings), in the same order as `importances`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """

    df = pd.DataFrame(columns=['importance'], index=features)
    for i, feature in enumerate(features):
        df.loc[feature, 'importance'] = importances[i]
    df = df.sort_values(by='importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = len(df)

    ax.barh(range(top_n), df['importance'], color='skyblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()  # Variable la plus importante en haut
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Variable importance', fontsize=14, pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()

    return fig
