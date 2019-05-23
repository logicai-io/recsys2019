import gc

import h5sparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from recsys.log_utils import get_logger
from recsys.utils import get_git_hash, timer
from scipy.special import logit

import os

from recsys.nn import nn_fit_predict

os.environ['OMP_NUM_THREADS'] = '1'
import gc

from multiprocessing.pool import ThreadPool

import h5sparse
import numpy as np
import pandas as pd
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.utils import timer, get_git_hash
from sklearn.metrics import roc_auc_score


logger = get_logger()

logger.info("Staring submission")

with timer("reading data"):
    meta = pd.read_hdf("../../data/proc/vectorizer_2/meta.h5", key="data")
    mat = h5sparse.File("../../data/proc/vectorizer_2/Xcsr.h5", mode="r")["matrix"]

with timer("splitting data"):
    train_ind = np.where(meta.is_test == 0)[0]
    val_ind = np.where(meta.is_test == 1)[0]
    logger.info(f"Train shape {train_ind.shape[0]} Val shape {val_ind.shape[0]}")
    meta_train = meta.iloc[train_ind]
    meta_val = meta.iloc[val_ind]
    X_train = mat[train_ind.min(): (train_ind.max() + 1)]
    X_val = mat[val_ind.min(): (val_ind.max() + 1)]
    del mat
    gc.collect()

with timer("model fitting"):
    n_cores = 60
    with ThreadPool(processes=n_cores) as pool:
        nn_preds = pool.starmap(nn_fit_predict, [((X_train, X_val), meta_train["was_clicked"].values)] * n_cores)
    val_pred = np.vstack(nn_preds).T.mean(axis=1)
    meta_val["click_proba"] = val_pred
    githash = get_git_hash()
    meta_val.to_csv(f"predictions/model_submit_clf_{githash}.csv", index=False)
