import random
import warnings
from functools import partial

import joblib
import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, hp, space_eval, tpe
from lightgbm import LGBMRanker
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1

# Define searched space
hyper_space = {'n_estimators': 100,
               'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
               'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
               'max_depth': hp.choice('max_depth', [4, 5, 8, -1]),
               'num_leaves': hp.choice('num_leaves', [15, 31, 63, 127]),
               'subsample': hp.uniform('subsample', 0.6, 1.0),
               'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)}

warnings.filterwarnings("ignore")

df = pd.read_csv("../../data/events_sorted_trans_all.csv", nrows=1000000)

df_train, df_val = split_by_timestamp(df)
vectorizer = make_vectorizer_1()

mat_train = vectorizer.fit_transform(df_train, df_train["was_clicked"])
print(mat_train.shape)
mat_val = vectorizer.transform(df_val)
print(mat_val.shape)


def objective(params):
    model = LGBMRanker()
    model.set_params(**params)
    model.fit(mat_train, df_train["was_clicked"], group=group_lengths(df_train["clickout_id"]))
    df_val["click_proba"] = model.predict(mat_val)
    mrr_val = mrr_fast(df_val, "click_proba")
    joblib.dump((params, mrr_val, df_val[["clickout_id", "item_id", "click_proba", "was_clicked"]]),
                "blend/" + str(random.randint(0,10000)) + ".joblib")
    return -mrr_val


trials = Trials()

# Set algoritm parameters
algo = partial(tpe.suggest,
               n_startup_jobs=20, gamma=0.25, n_EI_candidates=24)

# Fit Tree Parzen Estimator
best_vals = fmin(objective, space=hyper_space,
                 algo=algo, max_evals=60, trials=trials,
                 verbose=True,
                 rstate=np.random.RandomState(seed=2018))

# Print best parameters
best_params = space_eval(hyper_space, best_vals)
print("BEST PARAMETERS: " + str(best_params))

# Print best CV score
scores = [-trial['result']['loss'] for trial in trials.trials]
print("BEST CV SCORE: " + str(np.max(scores)))

records = []
for trial in trials.trials:
    obs = {}
    for key in trial['misc']['vals']:
        obs[key] = trial['misc']['vals'][key]
    obs['loss'] = trial['result']['loss']
    print(obs)
    records.append(obs)

df_hyperopt = pd.DataFrame.from_records(records)
df_hyperopt.to_csv("hyperopt_run.csv", index=False)

# scaler = make_pipeline(SanitizeSparseMatrix(), StandardScaler(with_mean=False))
# mat_train_s = scaler.fit_transform(mat_train, df_train["was_clicked"])
# mat_val_s = scaler.transform(mat_val)
#
# model_in = ks.Input(shape=(mat_train.shape[1],), dtype='float32', sparse=True)
# out = ks.layers.Dense(512, activation='relu')(model_in)
# out = ks.layers.Dense(256, activation='relu')(out)
# out = ks.layers.Dense(64, activation='relu')(out)
# out = ks.layers.Dense(1)(out)
# model = ks.Model(model_in, out)
# model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=0.001))
# model.fit(mat_train_s, df_train["was_clicked"], batch_size=256, epochs=3)
#
# df_train["click_proba_mlp"] = model.predict(mat_train_s)
# df_val["click_proba_mlp"] = model.predict(mat_val_s)
#
# print(mrr_fast(df_train, "click_proba_mlp"))
# print(mrr_fast(df_val, "click_proba_mlp"))
