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
pd.options.display.max_columns = 2000

events = pd.read_csv("../data/events_sorted.csv")
validation_results = pd.read_csv("../src/recsys/validation_results.csv")

events_trans = pd.read_csv("../data/events_sorted_trans_all.csv")

metadata = pd.read_csv("../data/item_metadata.csv")
metadata = dict(zip(metadata["item_id"], metadata["properties"]))

print("History")
session_id = "b4a04b7112f69"
print(events[events["session_id"]==session_id][["user_id", "action_type", "reference", "current_filters"]])
impressions = events[events["session_id"]==session_id]["impressions"].values[-1].split("|")
prices = events[events["session_id"]==session_id]["prices"].values[-1].split("|")
last_reference = events[events["session_id"]==session_id]["reference"].values[-1]
print()
print("Impressions")
for i, (item_id, price) in enumerate(zip(impressions, prices)):
    print(i, item_id, price, item_id==last_reference) #, metadata[int(item_id)])

# +
events_trans["price_rank"] = events_trans.groupby("clickout_id")["price"].rank("max", ascending=False)

events_trans[(events_trans["session_id"]==session_id)].to_csv("case.csv")
# -

validation_results[validation_results["session_id"]=="297b1a8701e3b"]

events_trans[events_trans["item_id"]==3367174]["price"]


