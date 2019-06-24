import gc

import h5sparse
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
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
    val_ind = np.where((meta.is_val == 1) & (meta.is_test == 0))[0]
    meta_val = meta.iloc[val_ind]
    X_val = mat[val_ind.min() : (val_ind.max() + 1)]

with timer("model fitting"):
    model: LGBMRanker = joblib.load("model_val.joblib")
    for n in range(100,1700,100):
        val_pred = model.predict(X_val, num_iteration=n)
        meta_val["click_proba"] = val_pred
        logger.info("Iter {} Val MRR {:.4f}".format(n, mrr_fast(meta_val, "click_proba")))
