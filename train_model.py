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

# ---- Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (
    RandomizedSearchCV, LeaveOneGroupOut, GridSearchCV)
from matplotlib.transforms import ScaledTranslation

import xgboost as xgb

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.modeling import plot_pred_vs_obs
from hdml.ml_helpers import plot_feature_importance

wtd_path = datadir / 'model' / "wtd_obs_training_dataset_sig4.csv"
if not wtd_path.exists():
    raise FileNotFoundError(
        "Make sure to run 'create_training_dataset.py' before running this "
        "script generate the 'wtd_obs_training_dataset.csv'."
        )


# %%

df = pd.read_csv(wtd_path)
df = df.dropna()

# grad -> slope
# hessian -> first derivative of the slope

# short -> stats over 7 pixels window == 210 m -> halfwidth de 105 m
# long -> stats over 41 pixels window = 1230 m -> halfwidth = 615 m
# stream -> stats along the line between point and nearest stream

features = [
    'dist_stream',
    'dist_top',
    # 'ratio_dist',
    'alt_stream',             # point_z - stream_z
    # 'alt_top',              # ridge_z - point_z
    'ratio_stream',         # (point_z - stream_z) / dist_stream
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
    'stream_grad_max',
    'stream_grad_var',
    'stream_grad_mean',
    'stream_hessian_max',
    'ndvi',
    'precipitation',
    'pre_mm_syr',  # average annual precipitation for sub-bassin
    'tmp_dc_syr',  # average annual air temperature for sub-bassin
    'pet_mm_syr',  # average annual potential evapotranspiration for sub-bassin
    'wetness_index',
    # 'point_z',      # Elevation of pixel
    'ndvi_yrly_avg',
    'precip_yrly_avg',
    'dist_divide',
    'alt_divide',
    'ratio_stream_divide'  # dist_stream / (dist_divide + dist_stream)
    # long dem stats (1230 m)
    'long_dem_max',
    'long_dem_mean',
    'long_dem_min',
    'long_dem_var',
    'long_dem_skew',
    'long_dem_kurt',
    # short dem stats (210 m)
    'short_dem_max',
    'short_dem_mean',
    'short_dem_min',
    'short_dem_var',
    'short_dem_skew',
    'short_dem_kurt',
    # stream stats
    'stream_dem_max',
    'stream_dem_mean',
    'stream_dem_min',
    'stream_dem_var',
    'stream_dem_skew',
    'stream_dem_kurt',
    ]


# %%

depths = df['NS'].values

std = np.std(depths)
mean = np.mean(depths)

# low_cutoff = mean - std
# high_cutoff = mean + std

# Percentile-based boundaries
low_cutoff = np.percentile(depths, 5)   # 10th percentile
high_cutoff = np.percentile(depths, 70)  # 90th percentile

df.loc[depths <= low_cutoff, 'NS_bin'] = 'shallow'
df.loc[depths >= high_cutoff, 'NS_bin'] = 'deep'
df.loc[(depths > low_cutoff) & (depths < high_cutoff), 'NS_bin'] = 'middle'

counts = df.NS_bin.value_counts()

counts_classes = [f'shallow\n<{low_cutoff: 0.1f} m',
                  f'middle\n] {low_cutoff:0.1f} - {high_cutoff: 0.1f} ] m',
                  f'deep\n>{high_cutoff: 0.1f} m']
counts_values = [counts.shallow, counts.middle, counts.deep]

fig, ax = plt.subplots(figsize=(5, 5))
bars = ax.bar(list(range(len(counts))), counts_values, color='skyblue')
ax.set_xlabel('Classes', fontsize=12, labelpad=10)
ax.set_ylabel('Nombre', fontsize=12, labelpad=10)
# ax.set_title("Histogramme des classes de milieu humide")
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.set_xticks(list(range(len(counts))))
ax.set_xticklabels(counts_classes)

# Ajout padding vertical (10 % au-dessus du max).
ax.set_ylim(top=counts.max() * 1.11)

# Ajout de la valeur et proportion au dessus de chaque barre.
ntot = len(df)
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


# %%

def greedy_permutation(coords):
    n = coords.shape[0]
    perm = [np.random.choice(n)]
    dists = np.full(n, np.inf)
    order = np.empty(n, dtype=int)
    order[perm[0]] = 0
    for i in range(1, n):
        d = np.linalg.norm(coords - coords[perm[-1]], axis=1)
        dists = np.minimum(dists, d)
        idx = np.argmax(dists)
        perm.append(idx)
        order[idx] = i
    return order


df_middle = df[df.NS_bin == 'middle'].copy()

coords = df_middle[['point_x', 'point_y']].values
order = greedy_permutation(coords)

df_middle['space_fill_order'] = order

df_middle = df_middle.sort_values(by='space_fill_order')

mask = ((df.NS_bin == 'shallow') | (df.NS_bin == 'deep'))
n_middle = np.sum(mask) // 2


index_to_keep = pd.Index(df_middle.index[:n_middle])
index_to_keep = index_to_keep.append(df.index[mask])

df_resample = df.loc[index_to_keep]
df_resample = df.copy()

# %%

fig2, ax2 = plt.subplots(figsize=(6, 4))

bins = list(range(0, 50, 1))

counts, bins, patches = ax2.hist(df_resample['NS'], bins, rwidth=0.8)

# %%

fig2, ax2 = plt.subplots(figsize=(6, 4))

bins = list(range(0, 50, 1))

counts, bins, patches = ax2.hist(df['NS'], bins, rwidth=0.8)

# # Annotate each bar.
# for count, bin_left, patch in zip(counts, bins, patches):
#     ax.text(
#         patch.get_x() + (patch.get_width()/2),
#         count,
#         str(int(count)),
#         ha='center',
#         va='bottom',
#     )

# # Add a little bit of space at the top of the graph for the text.
# ylim = ax.get_ylim()
# height_pixels = ax.get_window_extent().height
# data_per_pixel = (ylim[1] - ylim[0]) / height_pixels
# ypad = 10 * data_per_pixel
# ax.set_ylim(ylim[0], ylim[1] + ypad)

# ax.yaxis.grid(which='major', color='0.85')
# ax.set_axisbelow(True)
# ax.set_xlabel('Decade', labelpad=15, fontsize=12)
# ax.set_ylabel('Frequency', labelpad=15, fontsize=12)
# fig.suptitle(f'Number of WL observations for {country}')

# ax.set_xticks(bins)

# fig.tight_layout()

# return fig

# %%

plt.close('all')

import seaborn as sns

corr_matrix = df_resample[features].corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.set_theme(style="ticks", font_scale=0.8)
sns.heatmap(
    corr_matrix, annot=True, cmap='coolwarm',
    ax=ax, cbar=False, annot_kws={"fontsize": 8})
fig.suptitle('Correlation Matrix')
fig.tight_layout()
plt.show()


# %%
# features_to_plot = [
#     'dist_stream',
#     # 'dist_top',
#     # 'ratio_dist',
#     'alt_stream',             # point_z - stream_z
#     # 'alt_top',              # ridge_z - point_z
#     # 'ratio_stream',         # (point_z - stream_z) / dist_stream
#     # 'long_hessian_max',
#     # 'long_hessian_mean',
#     # 'long_hessian_var',
#     # 'long_hessian_skew',
#     # 'long_hessian_kurt',
#     # 'long_grad_mean',
#     # 'long_grad_var',
#     # 'short_grad_max',
#     # 'short_grad_var',
#     # 'short_grad_mean',
#     # 'stream_grad_max',
#     # 'stream_grad_var',
#     # 'stream_grad_mean',
#     # 'stream_hessian_max',
#     'ndvi',
#     'precipitation',
#     'pre_mm_syr',  # average annual precipitation for sub-bassin
#     'tmp_dc_syr',  # average annual air temperature for sub-bassin
#     'pet_mm_syr',  # average annual potential evapotranspiration for sub-bassin
#     'stream_to_total_dist_ratio',  # dist_stream / (dist_stream + dist_top)
#     'wetness_index',
#     # 'point_z'      # Elevation of pixel
#     'NS'
#     ]

# sns.pairplot(df_resample.loc[:, features_to_plot], hue='NS')
# %%
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

plt.close('all')

# 'Burkina', 'Guinee', 'Benin', 'Mali', 'Chad', 'Niger', 'Togo'
test_country = 'Mali'


train_index = df_resample[(df_resample.country == test_country)& (df_resample.LAT<11.5)].index
test_index = df_resample[(df_resample.country == test_country) & (df_resample.LAT>11.5)].index

# train_index = df_resample[(df_resample.country != test_country)].index
# test_index = df_resample[(df_resample.country == test_country)].index

df_train = df_resample.loc[train_index]
df_test = df_resample.loc[test_index]

fig2, ax2 = plt.subplots()
ax2.plot(df_train.LON, df_train.LAT, '.', color='orange')
ax2.plot(df_test.LON, df_test.LAT, '.', color='blue')
fig2.tight_layout()


X_train = df_resample.loc[train_index, features].values
y_train = (df_resample.loc[train_index, 'point_z'].values -
           df_resample.loc[train_index, 'NS'].values)


X_test = df_resample.loc[test_index, features].values
y_test = df_resample.loc[test_index, 'point_z'].values - df_resample.loc[test_index, 'NS'].values

# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)

# params = {
#     'random_state': 42,
#     'n_estimators': 100,
#     # 'max_depth': 10,
#     # 'subsample': 0.3,
#     # 'gamma': 1,
#     # 'colsample_bytree': 1,
#     # 'reg_alpha': 0,
#     # 'reg_lambda': 6,
#     # 'learning_rate': 0.25
#     }

params = {'subsample': 0.6,
          'reg_lambda': 2.5,
          'reg_alpha': 1.5,
          'n_estimators': 1000,
          'max_depth': 4,
          'learning_rate': 0.03,
          'gamma': 0.1,
          'colsample_bytree': 0.7}

Cl = xgb_model = xgb.XGBRegressor(**params)
# Cl = HistGradientBoostingRegressor(max_depth=3, loss='gamma', quantile=0.75)
# Cl = KNeighborsRegressor(n_neighbors=5)
# Cl = RandomForestRegressor(max_depth=4, n_estimators=1000)
Cl.fit(X_train, y_train)

# fig3 = plot_feature_importance(Cl.feature_importances_, features)

y_eval = Cl.predict(X_test)
classes = np.full(len(y_test), f'{test_country}')
axis = {'xmin': 300, 'xmax': 400, 'ymin': 300, 'ymax': 400}
fig4 = plot_pred_vs_obs(
    y_test, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )
fig4.tight_layout()

y_eval = Cl.predict(X_train)
classes = np.full(len(y_eval), f'All but {test_country}')
fig5 = plot_pred_vs_obs(
    y_train, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )


# %%
from sklearn.feature_selection import SequentialFeatureSelector

knn = KNeighborsRegressor(n_neighbors=10)

# Sélectionner les 3 meilleures caractéristiques en partant de la totalité (Backward)
sfs = SequentialFeatureSelector(knn, n_features_to_select=10, direction='backward')

sfs.fit(X_train, y_train)

print("\nLes 5 caractéristiques retenues par SBS :")
# On récupère les indices des colonnes sélectionnées
selected_features = np.array(features)[sfs.get_support()]
print(selected_features)
# %%10

point = df.loc[df.ID == 'KR-0432-F']

# %%

# Hyperparameter optimization (RandomizedSearchCV).

params_grid = {
    'n_estimators': [100, 500, 1000, 3000],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.5, 1],
    'reg_alpha': [0, 0.1, 1, 1.5, 2],
    'reg_lambda': [1, 1.5, 2, 2.5, 3]
    }

logo = LeaveOneGroupOut()

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=params_grid,
    n_iter=1000,
    scoring='neg_mean_squared_error',
    cv=logo,
    verbose=2,
    random_state=42,
    n_jobs=-1
    )


X = df_resample.loc[:, features]
y = df_resample.loc[:, 'NS']

random_search.fit(X, y, groups=df_resample.country)

print()
print("Best hyperparameters found :")
print(random_search.best_params_)

# %%

# Hyperparameter optimization (GridSearchCV).

Cl = RandomForestRegressor(random_state=42)

params_grid = {
    'n_estimators': [75, 100, 125],
    'max_depth': [8, 9, 10],
    'learning_rate': [0.05, 0.01, 0.02],
    'subsample': [0.5, 0.6, 0.7],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0.25, 0.5, 0.75],
    'reg_alpha': [1.75, 2, 2.25],
    'reg_lambda': [1.25, 1.5, 1.75]
    }

grid_search = GridSearchCV(
    estimator=Cl,
    param_grid=params_grid,
    scoring='precision_weighted',
    # cv=LeaveOneGroupOut(),
    verbose=2,
    n_jobs=-1
    )

grid_search.fit(X, y)

best_params = random_search.best_params_

print()
print("Best hyperparamter found :")
print(best_params)
