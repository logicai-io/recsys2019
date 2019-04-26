from collections import defaultdict

import numpy as np


def calculate_mean_rec_err(queries, clicks):
    recs = []
    for click, query in zip(clicks, queries):
        if query and click in query:
            rec = 1 / (query.index(click) + 1)
            rec = min(1, rec)
            recs.append(rec)
    return np.array(recs).mean()


def group_clickouts_into_list(df, predcol, append_index=False):
    sessions_items = defaultdict(list)
    df = df.sort_values(predcol, ascending=False)
    for id, (clickout_id, item_id) in enumerate(zip(df.clickout_id, df.item_id)):
        if append_index:
            sessions_items[clickout_id].append(id)
        else:
            sessions_items[clickout_id].append(item_id)
    return sessions_items


def mrr_fast(df_val, predcol):
    sessions_items = group_clickouts_into_list(df_val, predcol)
    val_check = df_val[df_val["was_clicked"] == 1][["clickout_id", "item_id"]]
    val_check["predicted"] = val_check["clickout_id"].map(sessions_items)
    return calculate_mean_rec_err(val_check["predicted"].tolist(), val_check["item_id"])
