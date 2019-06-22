import gc

import h5sparse
import joblib
import numpy as np
import pandas as pd
from recsys.metric import mrr_fast
from recsys.utils import group_lengths, timer
from sklearn.metrics import roc_auc_score


def run_model(mat_path, meta_path, model_instance, predictions_path, model_path, val, logger):
    with timer("read data"):
        meta = pd.read_hdf(meta_path, key="data")
        mat = h5sparse.File(mat_path, mode="r")["matrix"]

    with timer("split data"):
        if val:
            train_ind = np.where((meta.is_val == 0) & (meta.is_test == 0))[0]
            val_ind = np.where((meta.is_val == 1) & (meta.is_test == 0))[0]
        else:
            train_ind = np.where(meta.is_test == 0)[0]
            val_ind = np.where(meta.is_test == 1)[0]

        logger.info(f"Train shape {train_ind.shape[0]} Val shape {val_ind.shape[0]}")
        meta_train = meta.iloc[train_ind]
        meta_val = meta.iloc[val_ind]
        X_train = mat[train_ind.min() : (train_ind.max() + 1)]
        X_val = mat[val_ind.min() : (val_ind.max() + 1)]
        del mat
        gc.collect()

    with timer("fit model"):
        model_instance.fit(
            X_train, meta_train["was_clicked"].values, group=group_lengths(meta_train["clickout_id"].values)
        )
        joblib.dump(model_instance, model_path)
        val_pred = model_instance.predict(X_val)
        train_pred = model_instance.predict(X_train)
        logger.info("Train AUC {:.4f}".format(roc_auc_score(meta_train["was_clicked"].values, train_pred)))
        if val:
            logger.info("Val AUC {:.4f}".format(roc_auc_score(meta_val["was_clicked"].values, val_pred)))
        meta_val["click_proba"] = val_pred
        if val:
            logger.info("Val MRR {:.4f}".format(mrr_fast(meta_val, "click_proba")))
        meta_val.to_csv(predictions_path, index=False)
