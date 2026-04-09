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
    # 'long_hessian_max',
    # 'long_hessian_mean',
    # 'long_hessian_var',
    # 'long_hessian_skew',
    # 'long_hessian_kurt',
    # 'long_grad_mean',
    # 'long_grad_var',
    # 'short_grad_max',
    # 'short_grad_var',
    # 'short_grad_mean',
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


def eval_mean_error(y_true, y_pred):
    me = np.mean(y_true - y_pred)
    return float(me)


# Split training and test set.

plt.close('all')

if TEST_COUNTRY is not None:
    df_resample = df.loc[df.country == TEST_COUNTRY].copy()
else:
    df_resample = df.copy()

df_resample = df_resample.reset_index(drop=True)

# Define features (X), target (y), and groups (HYBAS_ID) for the split.
X = df_resample[FEATURES]
y = df_resample['NS']


from hdml.ml_helpers import plot_ns_distribution
plot_ns_distribution(y.values)

RANDOM_SPLIT = True

if TEST_COUNTRY == 'Mali' and RANDOM_SPLIT is False:
    train_idx = df_resample[df_resample.LON < -6].index.astype(int)
    test_idx = df_resample[df_resample.LON > -6].index.astype(int)
else:
    # Grouped split by watershed (20% of watersheds for the test set).
    groups = df_resample['HYBAS_ID']
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
        )
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

params = {
    'subsample': 0.8,
    'colsample_bytree': 0.75,

    'reg_lambda': 2.0,
    'reg_alpha': 0.5,

    'max_depth': 5,

    'n_estimators': 150,
    'learning_rate': 0.05,
    'gamma': 0.1,

    'eval_metric': eval_mean_error,
    }


weights = np.ones(len(y_train))
weights[y_train > 10] = 2

Cl = xgb_model = xgb.XGBRegressor(**params)
Cl.fit(X_train, y_train,
       sample_weight=weights,
       eval_set=[(X_train, y_train), (X_test, y_test)],
       verbose=False,
       )

# =============================================================================
# 3. Graphique : RMSE (Axe Gauche) vs Mean Error (Axe Droit)
# =============================================================================
results = Cl.evals_result()

fig_lc, ax_lc = plt.subplots(figsize=(10, 6))

obj_name = list(results['validation_0'].keys())[0]

# --- AXE PRINCIPAL (Gauche) : RMSE ---
line1 = ax_lc.plot(results['validation_0'][obj_name],
                   label='Train (RMSE)', color='orange', linestyle='-')
line2 = ax_lc.plot(results['validation_1'][obj_name],
                   label='Test (RMSE)', color='blue', linestyle='-')
ax_lc.set_xlabel("Nombre d'arbres (Itérations)")
ax_lc.set_ylabel('RMSE (Précision globale)', color='black')
ax_lc.tick_params(axis='y', labelcolor='black')

# --- AXE SECONDAIRE (Droite) : Mean Error (Biais) ---
ax_me = ax_lc.twinx()

# ON UTILISE LE NOM DE LA FONCTION COMME CLÉ ICI :
line3 = ax_me.plot(results['validation_0']['eval_mean_error'],
                   label='Train (Mean Error)', color='orange', linestyle=':')
line4 = ax_me.plot(results['validation_1']['eval_mean_error'],
                   label='Test (Mean Error)', color='blue', linestyle=':')

ax_me.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Zéro Biais')

ax_me.set_ylabel('Mean Error (Biais : Pred - Obs)', color='dimgrey')
ax_me.tick_params(axis='y', labelcolor='dimgrey')

ax_lc.set_title('Learning Curves : RMSE (précision) vs Mean Error (biais)')

lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax_lc.legend(lines, labels, loc='center right')

fig_lc.tight_layout()
plt.show()

if hasattr(Cl, 'best_iteration'):
    print('Best Iteration:', Cl.best_iteration)

# Check feature importances and validate model fit.
if MODELTYPE == 'xgboost':
    fig3 = plot_feature_importance(Cl.feature_importances_, FEATURES)

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

y_eval = Cl.predict(X_train)

classes = np.full(len(y_eval), 'All countries (train)')
fig5 = plot_pred_vs_obs(
    y_train, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )


# %%

# Train model on the entire dataset and save model to disk.

X_train = df[FEATURES]
y_train = df['NS']

if MODELTYPE == 'xgboost':
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
