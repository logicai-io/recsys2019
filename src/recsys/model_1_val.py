import gc

import h5sparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from recsys.metric import mrr_fast
from recsys.utils import timer
from sklearn.metrics import roc_auc_score

with timer('reading data'):
    meta = pd.read_hdf("../../data/events_sorted_trans_chunks/vectorizer_1/events_sorted_trans.h5", key="data")
    mat = h5sparse.File("../../data/events_sorted_trans_chunks/vectorizer_1/events_sorted_trans_features.h5", mode="r")[
              'matrix'][:]

with timer('splitting data'):
    train_ind = np.where((meta.is_val == 0) & (meta.is_test == 0))[0]
    val_ind = np.where((meta.is_val == 1) & (meta.is_test == 0))[0]
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
    train_pred = model.predict_proba(X_train)[:, 1]
    print("Train AUC {:.4f}".format(roc_auc_score(meta_train["was_clicked"].values, train_pred)))
    print("Val AUC {:.4f}".format(roc_auc_score(meta_val["was_clicked"].values, val_pred)))
    meta_val["click_proba"] = val_pred
    print("Val MRR {:.4f}".format(mrr_fast(meta_val, "click_proba")))
    meta_val.to_csv("predictions/model_1_val.csv", index=False)
