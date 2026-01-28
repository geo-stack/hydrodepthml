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

# ---- Standard imports
import os
from pathlib import Path

# ---- Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Local imports.
from hdml import CONF


"""
ml_helpers.py

Helper utilities for Machine Learning training and evaluation.
"""


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
