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
        "Make sure to run 'create_training_dataset.py' before running this "
        "script generate the 'wtd_obs_training_dataset.csv'."
        )

# %%

df = pd.read_csv(wtd_path)
df = df.dropna()

features = [
    'dist_stream',
    'dist_top',
    'ratio_dist',
    'alt_stream',
    'alt_top',
    'ratio_stream',
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
