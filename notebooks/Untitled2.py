# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import joblib
from recsys.metric import mrr_fast, mrr_fast_v2

df = pd.read_csv("../data/events_sorted_trans_all.csv", nrows=200000)

# imm = joblib.load("../data/item_metadata_map.joblib") 
item_metadata = pd.read_csv("../data/item_metadata.csv")

df = pd.merge(df, item_metadata, on="item_id", how="left")

df["properties_list"] = df["properties"].fillna("").map(lambda x: set(x.split("|")))
df["current_filters"] = df["current_filters"].fillna("").map(lambda x: set(x.split("|")))

df[df["current_filters"] == "Sort by Price|Hotel|Resort"][["user_id", "item_id", "price"]]

df["common_search"] = [len(row["current_filters"] & row["properties_list"]) for i,row in df.iterrows()]

mrr_fast(df[df["current_filters"]!=""], "common_search")

df[df["current_filters"]!=""]

df["user_id"].map(lambda x: x[0]).value_counts()


