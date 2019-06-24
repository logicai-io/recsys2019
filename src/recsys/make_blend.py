import glob
import hashlib
from multiprocessing.pool import Pool

import pandas as pd
from recsys.metric import mrr_fast, mrr_fast_v2
from recsys.submission import group_clickouts
from scipy.optimize import fmin_powell, fmin


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
        [(hsh, fn.replace("config.json", "predictions.csv")) for hsh, fn in zip(pred1_vals_hashes, pred1_vals) if
         hsh in common_hashes])
    pred1_subs_c = sorted(
        [(hsh, fn.replace("config.json", "predictions.csv")) for hsh, fn in zip(pred1_subs_hashes, pred1_subs) if
         hsh in common_hashes])
    return pred1_vals_c, pred1_subs_c


def read_prediction_val(fn):
    p = pd.read_csv(fn)
    p.sort_values(["user_id", "session_id", "step"], inplace=True)
    p.reset_index(inplace=True, drop=True)
    mrr = mrr_fast(p, "click_proba")
    return mrr, p


def read_prediction(fn):
    p = pd.read_csv(fn)
    p.sort_values(["user_id", "session_id", "step"], inplace=True)
    p.reset_index(inplace=True, drop=True)
    return p


if __name__ == '__main__':

    preds1_vals, preds1_subs = get_preds_1()
    preds2_vals, preds2_subs = get_preds_2()

    preds_vals_all = preds1_vals + preds2_vals
    preds_subs_all = preds1_subs + preds2_subs

    # read validation models
    with Pool(32) as pool:
        val_predictions_dfs = pool.map(read_prediction_val, [fn for _,fn in preds_vals_all])
    val_predictions = [(mrr, hsh, df) for ((hsh, fn), (mrr, df)) in zip(preds_vals_all, val_predictions_dfs) if (df.shape[0] == 3077674) and (mrr > 0.68)]
    val_hashes = [p[1] for p in val_predictions]

    # read submission models
    with Pool(32) as pool:
        sub_predictions_dfs = pool.map(read_prediction, [fn for _, fn in preds_subs_all])
    sub_predictions = [(hsh, df) for ((hsh, fn), df) in zip(preds_subs_all, sub_predictions_dfs) if hsh in val_hashes]

    final = val_predictions[-1][2].copy()

    def opt(v):
        final["click_proba"] = 0
        for c, (_, _, pred) in zip(v, val_predictions):
            # c = max(c,0)
            final["click_proba"] += c * pred["click_proba"]
        mrr = mrr_fast_v2(final["was_clicked"], final["click_proba"], final["clickout_id"])
        print(v)
        print(f"MRR {mrr}")
        return -mrr

    coefs = fmin(opt, [0] * len(val_predictions))
    coefs = fmin(opt, coefs)
    mrr = mrr_fast(final, "click_proba")
    mrr_str = f"{mrr:.4f}"[2:]
    print(mrr)

    final = sub_predictions[-1][1].copy()
    final["click_proba"] = 0
    for c, (hsh, pred) in zip(coefs, sub_predictions):
        # c = max(c,0)
        print(hsh, c*10000)
        final["click_proba"] += c * pred["click_proba"]
    _, submission_df = group_clickouts(final)
    save_as = f"submissions/submission_{mrr_str}.csv"
    print(f"Saving submission file {save_as}")
    submission_df.to_csv(save_as, index=False)
