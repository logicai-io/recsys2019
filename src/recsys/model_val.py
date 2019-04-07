import gc

import h5sparse
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.utils import group_lengths, timer
from sklearn.metrics import roc_auc_score

logger = get_logger()

print("Staring validation")

with timer("reading data"):
    meta = pd.read_hdf("../../data/proc/vectorizer_1/meta.h5", key="data")
    mat = h5sparse.File("../../data/proc/vectorizer_1/Xcsr.h5", mode="r")["matrix"][:]

with timer("splitting data"):
    train_ind = np.where((meta.is_val == 0) & (meta.is_test == 0))[0]
    val_ind = np.where((meta.is_val == 1) & (meta.is_test == 0))[0]
    logger.info(f"Train shape {train_ind.shape[0]} Val shape {val_ind.shape[0]}")
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
    train_pred = model.predict(X_train)
    logger.info("Train AUC {:.4f}".format(roc_auc_score(meta_train["was_clicked"].values, train_pred)))
    logger.info("Val AUC {:.4f}".format(roc_auc_score(meta_val["was_clicked"].values, val_pred)))
    meta_val["click_proba"] = val_pred
    logger.info("Val MRR {:.4f}".format(mrr_fast(meta_val, "click_proba")))
    meta_val.to_csv("predictions/model_2_val.csv", index=False)
