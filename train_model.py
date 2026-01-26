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
from sklearn.model_selection import (
    RandomizedSearchCV, LeaveOneGroupOut, GridSearchCV)

# ---- Local imports
from hdml import __datadir__ as datadir
from hdml.modeling import perform_cross_validation, plot_pred_vs_obs

model_path = datadir / 'model' / 'wtd_predict_model.pkl'


wtd_path = pd.read_csv(datadir / 'model' / "wtd_obs_training_dataset.csv")
if not wtd_path.exists():
    raise FileNotFoundError(
        "Make sure to run 'create_training_dataset.py' before running this "
        "script generate the 'wtd_obs_training_dataset.csv'."
        )


# %%

df = pd.read_csv(wtd_path)
df = df.dropna()

features = [
    'ratio_dist',
    'ratio_stream',
    'dist_stream',
    'alt_stream',
    'dist_top',
    'alt_top',
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
    ]


# %%

train_index = df[df.country != 'Mali'].index
test_index = df[df.country == 'Mali'].index

df_train = df.loc[train_index]
df_test = df.loc[test_index]

plt.plot(df_train.LON, df_train.LAT, '.', color='orange')
plt.plot(df_test.LON, df_test.LAT, '.', color='blue')
plt.tight_layout()


# %%

plt.close('all')

X = df[features].values
y = df['NS'].values

X_train = df.loc[train_index, features].values
y_train = df.loc[train_index, 'NS'].values

X_test = df.loc[test_index, features].values
y_test = df.loc[test_index, 'NS'].values

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

y_eval = Cl.predict(X_test)

importances = pd.DataFrame(columns=['importance'], index=features)
for f in range(len(features)):
    importances.loc[
        features[f], 'importance'
        ] = Cl.feature_importances_[f]
importances = importances.sort_values(by='importance', ascending=False)

print(importances)

classes = np.full(len(y_test), 'Mali')
axis = {'xmin': 0, 'xmax': 30, 'ymin': 0, 'ymax': 30}
fig = plot_pred_vs_obs(
    y_test, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )

y_eval = Cl.predict(X_train)
classes = np.full(len(y_eval), 'Mali')
fig2 = plot_pred_vs_obs(
    y_train, y_eval, classes, axis=axis,
    suptitle='True vs Predicted values',
    plot_stats=True
    )


# %%

# Hyperparameter optimization (RandomizedSearchCV).

Cl = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200, 300],         # Number of trees in the forest
    "max_depth": [3, 5, 8, 12, 20],              # Maximum depth of each tree
    "min_samples_split": [2, 4, 8, 12],          # Min samples required to split an internal node
    "min_samples_leaf": [1, 2, 4, 8],            # Min samples needed at a leaf node
    "max_features": ["sqrt", "log2", 0.5, 0.8],  # Number of features to consider at each split
    "max_samples": [0.5, 0.7, 0.9],              # Fraction of samples for each tree
    }

random_search = RandomizedSearchCV(
    estimator=Cl,
    param_distributions=param_grid,
    n_iter=100,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=0,
    n_jobs=-1
    )
random_search.fit(X, y)


best_params = random_search.best_params_

print()
print("Best hyperparamter found :")
print(best_params)

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
    scoring='neg_mean_squared_error',
    # cv=LeaveOneGroupOut(),
    verbose=2,
    n_jobs=-1
    )

grid_search.fit(X, y)

best_params = random_search.best_params_

print()
print("Best hyperparamter found :")
print(best_params)
