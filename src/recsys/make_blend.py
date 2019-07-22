import glob
import hashlib
import os
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from recsys.metric import mrr_fast
from recsys.mrr import mrr_fast_v3
from recsys.submission import group_clickouts
from recsys.utils import group_lengths
from scipy.optimize import fmin


def str_to_hash(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


# we have 2 predictions sets
def get_preds_1():
    """
    This set of predictions is identified by hash and a suffix with a model type
    :return:
    """
    pred1_vals = glob.glob("predictions/model_val_*.csv")
    pred1_vals_hashes = [fn.split("/")[-1].replace(".csv", "").replace("model_val_", "") for fn in pred1_vals]
    pred1_subs = glob.glob("predictions/model_submit_*.csv")
    pred1_subs_hashes = [fn.split("/")[-1].replace(".csv", "").replace("model_submit_", "") for fn in pred1_subs]
    common_hashes = set(pred1_subs_hashes) & set(pred1_vals_hashes)
    pred1_vals_c = sorted([(hsh, fn) for hsh, fn in zip(pred1_vals_hashes, pred1_vals) if hsh in common_hashes])
    pred1_subs_c = sorted([(hsh, fn) for hsh, fn in zip(pred1_subs_hashes, pred1_subs) if hsh in common_hashes])
    return pred1_vals_c, pred1_subs_c


def get_preds_2():
    """
    This set of predictions is identified by hash and a suffix with a model type
    :return:
    """
    pred1_vals = glob.glob("predictions/runs/*_val/config.json")
    pred1_vals_hashes = [fn.split("/")[-2].split("_")[0] for fn in pred1_vals]
    pred1_subs = glob.glob("predictions/runs/*_sub/config.json")
    pred1_subs_hashes = [fn.split("/")[-2].split("_")[0] for fn in pred1_subs]
    common_hashes = set(pred1_subs_hashes) & set(pred1_vals_hashes)
    pred1_vals_c = sorted(
        [
            (hsh, fn.replace("config.json", "predictions.csv"))
            for hsh, fn in zip(pred1_vals_hashes, pred1_vals)
            if hsh in common_hashes
        ]
    )
    pred1_subs_c = sorted(
        [
            (hsh, fn.replace("config.json", "predictions.csv"))
            for hsh, fn in zip(pred1_subs_hashes, pred1_subs)
            if hsh in common_hashes
        ]
    )
    return pred1_vals_c, pred1_subs_c


def read_prediction_val(fn):
    p = pd.read_csv(fn)
    p.sort_values(["user_id", "session_id", "step"], inplace=True)
    p.reset_index(inplace=True, drop=True)
    mrr = mrr_fast(p, "click_proba")
    config_file = fn.replace("predictions.csv", "config.json")
    if os.path.exists(config_file) and config_file.endswith("config.json"):
        config = open(config_file).read()
    else:
        config = fn
    return mrr, p, config


def read_prediction(fn):
    p = pd.read_csv(fn)
    p.sort_values(["user_id", "session_id", "step"], inplace=True)
    p.reset_index(inplace=True, drop=True)
    return p


if __name__ == "__main__":
    preds1_vals, preds1_subs = get_preds_1()
    preds2_vals, preds2_subs = get_preds_2()

    preds_vals_all = preds1_vals + preds2_vals
    preds_subs_all = preds1_subs + preds2_subs

    # read validation models
    with Pool(32) as pool:
        val_predictions_dfs = pool.map(read_prediction_val, [fn for _, fn in preds_vals_all])
    val_predictions = [
        (mrr, hsh, df, config)
        for ((hsh, fn), (mrr, df, config)) in zip(preds_vals_all, val_predictions_dfs)
        if (df.shape[0] == 3_077_674) and (mrr > 0.68) and ("160357" not in fn) and ("59629" not in fn)
    ]
    val_hashes = [p[1] for p in val_predictions]

    print("Debuging click probas")
    for mrr, hsh, df, _ in val_predictions:
        print(mrr, hsh, df["click_proba"].min(), df["click_proba"].max())

    final = val_predictions[-1][2].copy()

    lengths = group_lengths(final["clickout_id"])
    preds_stack = np.vstack([df["click_proba"] for _, _, df, _ in val_predictions]).T

    def opt(v):
        preds_ens = preds_stack.dot(v)
        mrr = mrr_fast_v3(final["was_clicked"].values, preds_ens, lengths)
        print(f"MRR {mrr}")
        return -mrr

    coefs = fmin(opt, [0] * preds_stack.shape[1])
    coefs = fmin(opt, coefs, ftol=0.000_001)

    final["click_proba"] = preds_stack.dot(coefs)
    mrr = mrr_fast(final, "click_proba")
    mrr_str = f"{mrr:.4f}"[2:]
    print(mrr)

    mrrs, _, _, configs = list(zip(*val_predictions))
    summary_df = pd.DataFrame({"config": configs, "mrr": mrrs, "coef": coefs})
    print(summary_df)
    summary_df.to_csv(f"model_summary_{mrr_str}.csv")

    # read submission models
    with Pool(32) as pool:
        sub_predictions_dfs = pool.map(read_prediction, [fn for _, fn in preds_subs_all])

    sub_predictions = [(hsh, df) for ((hsh, fn), df) in zip(preds_subs_all, sub_predictions_dfs) if hsh in val_hashes]
    for coef, (hsh, df) in zip(coefs, sub_predictions):
        print(coef, hsh, df["click_proba"].min(), df["click_proba"].max())
    sub_preds_stack = np.vstack([df["click_proba"] for _, df in sub_predictions]).T
    final = sub_predictions[-1][1].copy()
    final["click_proba"] = sub_preds_stack.dot(coefs)
    _, submission_df = group_clickouts(final)
    save_as = f"submissions/submission_{mrr_str}.csv"
    print(f"Saving submission file {save_as}")
    submission_df.to_csv(save_as, index=False)
