import gc

import h5sparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from recsys.utils import timer

with timer('reading data'):
    meta = pd.read_hdf("../../data/events_sorted_trans_chunks/vectorizer_1/events_sorted_trans.h5", key="data")
    mat = h5sparse.File("../../data/events_sorted_trans_chunks/vectorizer_1/events_sorted_trans_features.h5", mode="r")[
              'matrix'][:]

with timer('splitting data'):
    train_ind = np.where(meta.is_test == 0)[0]
    val_ind = np.where(meta.is_test == 1)[0]
    print(f"Train shape {train_ind.shape[0]} Val shape {val_ind.shape[0]}")
    meta_train = meta.iloc[train_ind]
    meta_val = meta.iloc[val_ind]
    X_train = mat[train_ind]
    X_val = mat[val_ind]
    del mat
    gc.collect()

with timer('model fitting'):
    model = LGBMClassifier(n_estimators=1600, n_jobs=-2)
    model.fit(X_train, meta_train["was_clicked"].values)
    val_pred = model.predict_proba(X_val)[:, 1]
    meta_val["click_proba"] = val_pred
    meta_val.to_csv("predictions/model_1_submit.csv", index=False)