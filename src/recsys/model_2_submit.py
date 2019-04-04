import gc

import h5sparse
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from recsys.utils import group_lengths, timer

with timer("reading data"):
    meta = pd.read_hdf("../../data/proc/vectorizer_1/meta.h5", key="data")
    mat = h5sparse.File("../../data/proc/vectorizer_1/Xcsr.h5", mode="r")["matrix"][:]

with timer("splitting data"):
    train_ind = np.where(meta.is_test == 0)[0]
    val_ind = np.where(meta.is_test == 1)[0]
    print(f"Train shape {train_ind.shape[0]} Val shape {val_ind.shape[0]}")
    meta_train = meta.iloc[train_ind]
    meta_val = meta.iloc[val_ind]
    X_train = mat[train_ind]
    X_val = mat[val_ind]
    del mat
    gc.collect()

with timer("model fitting"):
    model = LGBMRanker(n_estimators=1600, num_leaves=62, n_jobs=-2)
    model.fit(X_train, meta_train["was_clicked"].values, group=group_lengths(meta_train["clickout_id"].values))
    val_pred = model.predict(X_val)
    meta_val["click_proba"] = val_pred
    meta_val.to_csv("predictions/model_2_submit.csv", index=False)
