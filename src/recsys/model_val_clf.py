import gc
from scipy.special import logit

import h5sparse
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMClassifier
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.utils import group_lengths, timer, get_git_hash
from sklearn.metrics import roc_auc_score

logger = get_logger()

print("Staring validation")

with timer("reading data"):
    meta = pd.read_hdf("../../data/proc/vectorizer_1/meta.h5", key="data")
    mat = h5sparse.File("../../data/proc/vectorizer_1/Xcsr.h5", mode="r")["matrix"]

with timer("splitting data"):
    train_ind = np.where((meta.is_val == 0) & (meta.is_test == 0))[0]
    val_ind = np.where((meta.is_val == 1) & (meta.is_test == 0))[0]
    logger.info(f"Train shape {train_ind.shape[0]} Val shape {val_ind.shape[0]}")
    meta_train = meta.iloc[train_ind]
    meta_val = meta.iloc[val_ind]
    X_train = mat[train_ind.min() : (train_ind.max() + 1)]
    X_val = mat[val_ind.min() : (val_ind.max() + 1)]
    del mat
    gc.collect()

with timer("model fitting"):
    model = LGBMClassifier(n_estimators=1600, num_leaves=62, n_jobs=-2)
    model.fit(X_train, meta_train["was_clicked"].values)
    val_pred = logit(model.predict_proba(X_val)[:, 1])
    train_pred = logit(model.predict_proba(X_train)[:, 1])
    logger.info("Train AUC {:.4f}".format(roc_auc_score(meta_train["was_clicked"].values, train_pred)))
    logger.info("Val AUC {:.4f}".format(roc_auc_score(meta_val["was_clicked"].values, val_pred)))
    meta_val["click_proba"] = val_pred
    logger.info("Val MRR {:.4f}".format(mrr_fast(meta_val, "click_proba")))
    githash = get_git_hash()
    meta_val.to_csv(f"predictions/model_val_clf_{githash}.csv", index=False)
