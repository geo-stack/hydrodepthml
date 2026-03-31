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

# ---- Standard imports
from datetime import datetime
import pickle

# ---- Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.modeling import plot_pred_vs_obs

model_path = datadir / 'model' / 'wtd_predict_model.pkl'

wtd_path = datadir / 'model' / "wtd_obs_training_dataset.csv"

if not wtd_path.exists():
    raise FileNotFoundError(
        "Make sure to run '08_create_training_dataset.py' before running "
        "this script to generate your training dataset."
        )

df = pd.read_csv(wtd_path)

# %%


# =============================================================================
# FEATURE DESCRIPTIONS
# =============================================================================
#
# Dependent Variable
# ------------------
# NS:
#     Observation of the depth of the water table in meters
#     below the ground surface.
#
# Topographic and Spatial Features
# --------------------------------
# elev:
#     Elevation in meters from the raw, unmodified NASADEM DEM.
# dist_stream:
#     Euclidean distance to the nearest stream pixel in meters.
#     dist_stream = ((point_x - stream_x)**2 + (point_y - stream_y)**2)**0.5
# alt_stream:
#     Elevation difference (m) from the point to the nearest stream.
#     alt_stream = point_z - stream_z
# ratio_stream:
#     Overall slope ratio towards the nearest stream.
#     ratio_stream = alt_stream / max(dist_stream, pixel_size)
# dist_divide:
#     Euclidean distance (m) to the nearest watershed boundary.
#     Boundaries are derived from D8 subbasins using streams.
#     dist_divide = ((point_x - divide_x)**2 + (point_y - divide_y)**2)**0.5
# alt_divide:
#     Elevation difference (m) from the nearest watershed boundary
#     to the observation point.
#     alt_divide = divide_z - point_z
# ratio_stream_divide:
#     Relative position of point between stream and the watershed boundary.
#     ratio = dist_stream / (dist_divide + dist_stream)
# wetness_index:
#     Topographic Wetness Index (TWI) representing soil moisture
#     accumulation patterns.
#
# Geomorphometric Statistics
# --------------------------
# long_dem_*:
#     Descriptive statistics of conditioned elevation (point_z) over a
#     41-pixel window (1230 m -> 615 m halfwidth).
# short_dem_*:
#     Descriptive statistics of conditioned elevation (point_z) over a
#     7-pixel window (210 m -> 105 m halfwidth).
# stream_dem_*:
#     Descriptive statistics of conditioned elevation (point_z) calculated
#     along a Bresenham line between the point and the nearest stream.
# long_grad_*:
#     Descriptive statistics of the slope (first derivative of elevation)
#     over a 41-pixel window (1230 m -> 615 m halfwidth).
# short_grad_*:
#     Descriptive statistics of the slope (first derivative of elevation)
#     over a 7-pixel window (210 m -> 105 m halfwidth).
# stream_grad_*:
#     Descriptive statistics of the slope (first derivative of elevation)
#     calculated along a Bresenham line between the point and the nearest
#     stream.
# long_hessian_*:
#     Descriptive statistics of the terrain curvature (second derivative
#     of elevation) over a 41-pixel window (1230 m -> 615 m halfwidth).
# stream_hessian_*:
#     Descriptive statistics of the terrain curvature (second derivative
#     of elevation) calculated along a Bresenham line between the point
#     and the nearest stream.
#
# Climatic and Environmental Features
# -----------------------------------
# ndvi:
#     Mean daily Normalized Difference Vegetation Index (NDVI) averaged
#     over a period of time before each specific observation. The length
#     of the averaged time window depends on the size of the watershed
#     containing the observation point.
# precipitation:
#     Mean daily precipitation (mm) averaged over a period of time before
#     each specific observation. The length of the averaged time window
#     depends on the size of the watershed containing the observation point.
# ndvi_yrly_avg:
#     Mean basin NDVI averaged over the 2 years prior to the observation date.
# precip_yrly_avg:
#     Mean basin precipitation (mm) averaged over the 2 years prior to
#     the observation date.
# pre_mm_syr:
#     Average annual precipitation (mm/year) for the sub-basin.
# tmp_dc_syr:
#     Average annual air temperature (°C) for the sub-basin.
# pet_mm_syr:
#     Average annual potential evapotranspiration (mm/year) for the sub-basin.
# =============================================================================

features = [
    # ---- TOPOGRAPHIC AND SPATIAL FEATURES
    'elev',
    # 'point_z',
    'dist_stream',
    'alt_stream',
    'ratio_stream',
    'dist_divide',
    'alt_divide',
    'ratio_stream_divide',
    # ---- GEOMORPHOMETRIC STATISTICS
    'long_hessian_max',
    'long_hessian_mean',
    'long_hessian_var',
    'long_hessian_skew',
    'long_hessian_kurt',
    'long_grad_mean',
    'long_grad_var',
    'short_grad_max',
    'short_grad_var',
    'short_grad_mean',
    'long_dem_max',
    'long_dem_mean',
    'long_dem_min',
    'long_dem_var',
    'long_dem_skew',
    'long_dem_kurt',
    'short_dem_max',
    'short_dem_mean',
    'short_dem_min',
    'short_dem_var',
    'short_dem_skew',
    'short_dem_kurt',
    # 'stream_grad_max',
    # 'stream_grad_var',
    # 'stream_grad_mean',
    # 'stream_hessian_max',
    # 'stream_dem_max',
    # 'stream_dem_mean',
    # 'stream_dem_min',
    # 'stream_dem_var',
    # 'stream_dem_skew',
    # 'stream_dem_kurt',
    # ---- CLIMATIC AND ENVIRONMENTAL FEATURES
    'ndvi',
    'precipitation',
    'pre_mm_syr',
    'tmp_dc_syr',
    'pet_mm_syr',
    'wetness_index',
    'ndvi_yrly_avg',
    'precip_yrly_avg',
    ]


# %%

X_train = df[features]
y_train = df['NS']

params = {
    'n_estimators': 50,
    'random_state': 42
    }

Cl = RandomForestRegressor(**params)
Cl.fit(X_train, y_train)

# Save the model.
model_data = {
    'model': Cl,
    'feature_names': features,
    'training_date': datetime.now().strftime('%Y-%m-%d')
    }
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

# %%

importances = pd.DataFrame(columns=['importance'], index=features)
for f in range(len(features)):
    importances.loc[
        features[f], 'importance'
        ] = Cl.feature_importances_[f]
importances = importances.sort_values(by='importance', ascending=False)

print(importances)

fig, ax = plt.subplots(figsize=(10, 8))
top_n = len(importances)

ax.barh(range(top_n), importances['importance'], color='skyblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(importances.index)
ax.invert_yaxis()  # Variable la plus importante en haut
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Variable importance', fontsize=14, pad=15)
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()

# %%

y_eval = Cl.predict(X_train)

y_max = max(np.max(y_eval), np.max(y_train))

classes = df.country.values
fig2 = plot_pred_vs_obs(
    y_train, y_eval, classes,
    axis={'xmin': 0, 'xmax': y_max, 'ymin': 0, 'ymax': y_max},
    suptitle='True vs Predicted values',
    plot_stats=False
    )
