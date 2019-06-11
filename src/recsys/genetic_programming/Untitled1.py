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

# +
import pandas as pd
import numpy as np
import datatable as dt
from tqdm import tqdm
import graphviz
from collections import defaultdict
from gplearn.functions import make_function

ACTIONS_WITH_ITEM_REFERENCE = {
    "search for item",
    "interaction item info",
    "interaction item image",
    "interaction item deals",
    "interaction item rating",
    "clickout item",
}
# -

df = dt.fread("../../../data/events_sorted.csv") 

ref = df[:,["user_id", "action_type", "session_id", "reference", "timestamp"]]

ref = ref.topandas()

for user_id in ["0029BRXGBS69"]:
    clickouts = ref[(ref["user_id"]==user_id)&(ref["action_type"]=="clickout item")]
    print(clickouts.shape[0], clickouts["reference"].nunique())
    print(clickouts)

clickouts_num = ref[ref["action_type"]=="clickout item"].groupby("user_id")["session_id"].count().sort_values(ascending=False)
clickouts_num.value_counts()


