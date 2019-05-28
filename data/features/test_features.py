import pandas as pd
import glob
from recsys.metric import mrr_fast_v2, mrr_fast

nrows = 2000000
meta = pd.read_csv("_meta_feautres_all.csv", nrows=nrows)
print(meta.head())
meta["revrank"] = -meta["rank"]
print("rank", mrr_fast(meta, "revrank"))
for fn in glob.glob("*.csv"):
    if fn.startswith("_meta"):
        continue
    if fn.startswith("graph"):
        continue
    fs = pd.read_csv(fn, nrows=nrows)
    for col in fs:
        if col == "clickout_id":
            continue
        meta["test"] = -fs[col] - meta["rank"]/100000
        meta["test_group"] = meta.groupby("clickout_id")["test"].rank("max", ascending=True)
        print(col, mrr_fast(meta, "test"), mrr_fast(meta, "test_group"))

"""
rank 0.4601196641849266
graph_similarity_user_item_random_walk 0.4499179564831023 0.4499179564831023
graph_similarity_user_item_clickout 0.4682862013914295 0.4682862013914295
graph_similarity_user_item_search 0.4597668339135057 0.4597668339135057
graph_similarity_user_item_interaction_info 0.4617315954882975 0.4617315954882975
graph_similarity_user_item_interaction_img 0.4757111710533586 0.4757111710533586
graph_similarity_user_item_intearction_deal 0.4610211642312579 0.4610211642312579
graph_similarity_user_item_all_interactions 0.49141947224442356 0.49141947224442356
graph_similarity_user_item_random_walk_resets 0.49249302942245154 0.49249302942245154
user_item_random_walk_sim 0.45423091518654884 0.45423091518654884
"""

df = pd.read_csv("../events_sorted_trans_all.csv", nrows=nrows)
for col in ["similar_users_item_interaction", "most_similar_item_interaction"]:
    if col == "clickout_id":
        continue
    meta["test"] = df[col] - meta["rank"] / 100000
    meta["test_group"] = meta.groupby("clickout_id")["test"].rank("max", ascending=True)
    print(col, mrr_fast(meta, "test"), mrr_fast(meta, "test_group"))

