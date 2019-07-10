import gc

import h5sparse
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from recsys.config import BEST_PARAMS
from recsys.log_utils import get_logger
from recsys.utils import group_lengths, timer, get_git_hash

logger = get_logger()

logger.info("Staring submission")

with timer("reading data"):
    meta = pd.read_hdf("../../data/proc/vectorizer_1/meta.h5", key="data")
    mat = h5sparse.File("../../data/proc/vectorizer_1/Xcsr.h5", mode="r")["matrix"]

with timer("splitting data"):
    train_ind = np.where(meta.is_test == 0)[0]
    val_ind = np.where(meta.is_test == 1)[0]
    logger.info(f"Train shape {train_ind.shape[0]} Val shape {val_ind.shape[0]}")
    meta_train = meta.iloc[train_ind]
    meta_val = meta.iloc[val_ind]
    X_train = mat[train_ind.min() : (train_ind.max() + 1)]
    X_val = mat[val_ind.min() : (val_ind.max() + 1)]
    del mat
    gc.collect()

with timer("model fitting"):
    model = LGBMRanker(**BEST_PARAMS)
    model.fit(X_train, meta_train["was_clicked"].values, group=group_lengths(meta_train["clickout_id"].values))
    val_pred = model.predict(X_val)
    meta_val["click_proba"] = val_pred
    githash = get_git_hash()
    meta_val.to_csv(f"predictions/model_submit_{githash}.csv", index=False)
    joblib.dump(model, "model_submit.joblib")
