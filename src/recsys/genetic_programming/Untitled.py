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

# %matplotlib inline
import pandas as pd

# +
import json
from collections import defaultdict

with open("click_indices.ndjson") as inp:
    lines = inp.readlines()

lines = list(map(json.loads, lines))

# +
from tqdm import tqdm
from collections import Counter


def group_time(t):
    if t <= 12:
        return t
    else:
        return int(t / 4) * 4


diff_dist = defaultdict(list)
records = []
for i, user in tqdm(enumerate(lines)):
    for session in user:
        if len(session) > 1:
            for ((t1, c1), (t2, c2)) in zip(session[:-1], session[1:]):
                if t2 - t1 <= 120:
                    records.append((c2 - c1, t2 - t1))
# -

df = pd.DataFrame.from_records(records)
df.columns = ["click_offset", "diff_time"]
df["diff_time_grouped"] = df["diff_time"].map(group_time)
df["one"] = 1

diff_time_pivot = df.pivot_table(values="one", index="diff_time_grouped", columns="click_offset", aggfunc=len)
diff_time_pivot.fillna(1.0, inplace=True)
diff_time_pivot = diff_time_pivot.div(diff_time_pivot.sum(axis=1), axis=0)
diff_time_pivot.to_csv("click_index_time_diff_pivot.csv")

# +
# unpivot

records = []
for ind in diff_time_pivot.index:
    for col in diff_time_pivot.columns:
        records.append((ind, col, diff_time_pivot.ix[ind, col]))
# -

diff_time = pd.DataFrame.from_records(records)
diff_time.columns = ["time", "offset", "prob"]
diff_time.to_csv("diff_click_offset.csv", sep="\t", index=False)

d = dict(zip(diff_time.apply(lambda row: (int(row["offset"]), int(row["time"])), axis=1), diff_time["prob"]))
import joblib

joblib.dump(d, "click_probs_by_index.joblib")

d

lines
