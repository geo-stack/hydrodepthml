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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import NuSVR
import xgboost as xgb

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.modeling import plot_pred_vs_obs
from hdml.ml_helpers import plot_feature_importance

model_path = datadir / 'model' / 'wtd_predict_model.pkl'

wtd_path = datadir / 'model' / 'wtd_obs_training_dataset_sig1_st500.csv'

if not wtd_path.exists():
    raise FileNotFoundError(
        "Make sure to run '08_create_training_dataset.py' before running "
        "this script to generate your training dataset."
        )

df = pd.read_csv(wtd_path)

MODELTYPE = 'xgboost'  # 'xgboost' or 'support_vector'

TEST_COUNTRY = [
    'Burkina',  # 0
    'Guinee',   # 1
    'Benin',    # 2
    'Mali',     # 3
    'Chad',     # 4
    'Niger',    # 5
    'Togo',     # 6
    None,       # 7
    ][3]

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
# pre_mm_syr:
#     Average annual precipitation (mm/year) for the sub-basin.
# tmp_dc_syr:
#     Average annual air temperature (°C) for the sub-basin.
# pet_mm_syr:
#     Average annual potential evapotranspiration (mm/year) for the sub-basin.
# =============================================================================

# List of features to use for training the model.
# Comment out features you do not want to use.

FEATURES = [
    # ---- TOPOGRAPHIC AND SPATIAL FEATURES
    'dist_stream',
    'alt_stream',
    'ratio_stream',
    'dist_divide',
    'alt_divide',
    'ratio_stream_divide',
    'wetness_index',
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
    # 'stream_grad_max',
    # 'stream_grad_var',
    # 'stream_grad_mean',
    # 'stream_hessian_max',
    # ---- CLIMATIC AND ENVIRONMENTAL FEATURES
    'ndvi',
    'precipitation',
    'pre_mm_syr',
    'tmp_dc_syr',
    'pet_mm_syr',
    ]


# %%

# Split training and test set.

plt.close('all')

df_resample = df.copy()

# Define features (X), target (y), and groups (HYBAS_ID) for the split.
X = df_resample[FEATURES]
y = df_resample['NS']

groups = df_resample['HYBAS_ID']

# Grouped split by watershed (20% of watersheds for the test set).
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

df_train = df_resample.iloc[train_idx]
df_test = df_resample.iloc[test_idx]

# Visualize the spatial distribution of the train and test sets.
fig2, ax2 = plt.subplots()
ax2.plot(df_train.LON, df_train.LAT, '.', color='orange', label='Train (80%)')
ax2.plot(df_test.LON, df_test.LAT, '.', color='blue', label='Test (20%)')
ax2.set_title("Spatial distribution (Split by HYBAS_ID)")
ax2.legend()
fig2.tight_layout()

# Extract NumPy arrays for model training.
X_train = X.iloc[train_idx].values
X_test = X.iloc[test_idx].values
y_train = y.iloc[train_idx].values
y_test = y.iloc[test_idx].values

if MODELTYPE == 'xgboost':
    params = {
        'subsample': 0.5,
        'reg_lambda': 0.1,
        'reg_alpha': 1.5,
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.1,
        'gamma': 0.2,
        'colsample_bytree': 0.9,
        }

    params = {
        'subsample': 0.7,
        'reg_lambda': 2.5,
        'reg_alpha': 1.5,
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.2,
        'gamma': 0.095,
        'colsample_bytree': 0.5
        }

    params = {
        'subsample': 0.8,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.05,
        'gamma': 0.0,
        'colsample_bytree': 0.8,
        }

    # params['objective'] = 'reg:absoluteerror'

    Cl = xgb_model = xgb.XGBRegressor(**params)
elif MODELTYPE == 'support_vector':
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    Cl = svr = NuSVR(C=50, nu=0.95)

LOG_TRANSFORM = True

if LOG_TRANSFORM:
    Cl.fit(X_train, np.log1p(y_train))
else:
    Cl.fit(X_train, y_train)

# Check feature importances and validate model fit.
if MODELTYPE == 'xgboost':
    fig3 = plot_feature_importance(Cl.feature_importances_, FEATURES)

if LOG_TRANSFORM:
    y_eval = np.expm1(Cl.predict(X_test))
else:
    y_eval = Cl.predict(X_test)

classes = np.full(len(y_test), 'All countries (test)')
axis = {'xmin': y_test.min(), 'xmax': y_test.max(),
        'ymin': y_test.min(), 'ymax': y_test.max()}
fig4 = plot_pred_vs_obs(
    y_test, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )
fig4.tight_layout()

if LOG_TRANSFORM:
    y_eval = np.expm1(Cl.predict(X_train))
else:
    y_eval = Cl.predict(X_train)

classes = np.full(len(y_eval), 'All countries (train)')
fig5 = plot_pred_vs_obs(
    y_train, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )

# %%
# keep_index = df[df.world_koppen == 3].index
# df_resample = df.loc[keep_index].copy()

df_resample = df.copy()

train_index = df_resample[(df_resample.country != TEST_COUNTRY)].index
test_index = df_resample[(df_resample.country == TEST_COUNTRY)].index

df_train = df_resample.loc[train_index]
df_test = df_resample.loc[test_index]

fig2, ax2 = plt.subplots()
ax2.plot(df_train.LON, df_train.LAT, '.', color='orange')
ax2.plot(df_test.LON, df_test.LAT, '.', color='blue')
fig2.tight_layout()

X_train = df_resample.loc[train_index, FEATURES].values
X_test = df_resample.loc[test_index, FEATURES].values
y_train = df_resample.loc[train_index, 'NS'].values
y_test = df_resample.loc[test_index, 'NS'].values

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)






# %%

from sklearn.feature_selection import SequentialFeatureSelector

knn = KNeighborsRegressor(n_neighbors=10)

# Sélectionner les 3 meilleures caractéristiques en partant
# de la totalité (Backward)
sfs = SequentialFeatureSelector(knn, n_features_to_select=10, direction='backward')

sfs.fit(X_train, y_train)

print("\nLes 5 caractéristiques retenues par SBS :")
# On récupère les indices des colonnes sélectionnées
selected_features = np.array(FEATURES)[sfs.get_support()]
print(selected_features)

# %%
# fig3 = plot_feature_importance(Cl.feature_importances_, features)

# y_eval = Cl.predict(X_test)
# classes = np.full(len(y_test), f'{test_country}')
# axis = {'xmin': y_test.min(), 'xmax': y_test.max(),
#         'ymin': y_test.min(), 'ymax': y_test.max()}
# fig4 = plot_pred_vs_obs(
#     y_test, y_eval, classes, axis=axis,
#     suptitle='True vs Predicted values',
#     plot_stats=True
#     )
# fig4.tight_layout()

# y_eval = Cl.predict(X_train)
# classes = np.full(len(y_eval), f'All but {test_country}')
# fig5 = plot_pred_vs_obs(
#     y_train, y_eval, classes, axis=axis,
#     suptitle='True vs Predicted values',
#     plot_stats=True
#     )


# %%

# Train model on the entire dataset and save model to disk.

X_train = df[FEATURES]
y_train = df['NS']

if MODELTYPE == 'xgboost':
    params = {'subsample': 0.5,
              'reg_lambda': 2.5,
              'reg_alpha': 1.5,
              'n_estimators': 150,
              'max_depth': 4,
              'learning_rate': 0.1,
              'gamma': 0.2,
              'colsample_bytree': 0.9}

    Cl = xgb_model = xgb.XGBRegressor(**params)
elif MODELTYPE == 'support_vector':
    Cl = svr = NuSVR(C=50, nu=0.95)

Cl.fit(X_train, y_train)

# Save the model.
model_data = {
    'model': Cl,
    'feature_names': FEATURES,
    'training_date': datetime.now().strftime('%Y-%m-%d')
    }
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

# %%

importances = pd.DataFrame(columns=['importance'], index=FEATURES)
for f in range(len(FEATURES)):
    importances.loc[
        FEATURES[f], 'importance'
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
