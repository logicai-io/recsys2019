from collections import defaultdict
from itertools import groupby

import numpy as np


def calculate_mean_rec_err(queries, clicks):
    recs = []
    for click, query in zip(clicks, queries):
        if query and click in query:
            rec = 1 / (query.index(click) + 1)
            rec = min(1, rec)
            recs.append(rec)
    return np.array(recs).mean()


def group_clickouts_into_list(df, predcol):
    sessions_items = defaultdict(list)
    df = df.sort_values(predcol, ascending=False)
    for clickout_id, item_id in zip(df.clickout_id, df.item_id):
        sessions_items[clickout_id].append(item_id)
    return sessions_items


def mrr_fast(df_val, predcol):
    sessions_items = group_clickouts_into_list(df_val, predcol)
    val_check = df_val[df_val["was_clicked"] == 1][["clickout_id", "item_id"]]
    val_check["predicted"] = val_check["clickout_id"].map(sessions_items)
    return calculate_mean_rec_err(val_check["predicted"].tolist(), val_check["item_id"])


def mrr_fast_v2(y, yhat, group):
    def get_mrr(ys):
        try:
            return 1 / (ys.index(1) + 1)
        except ValueError:
            return 1 / len(ys)

    grouped = list(zip(y, yhat, group))
    mrr_sum = 0
    n = 0
    for g, items in groupby(grouped, lambda x: x[2]):
        items_sorted = sorted(items, key=lambda x: -x[1])
        ys, _, _ = list(zip(*items_sorted))
        mrr_sum += get_mrr(ys)
        n += 1
    return mrr_sum / n
