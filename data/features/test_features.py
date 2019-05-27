import pandas as pd
import glob
from recsys.metric import mrr_fast_v2, mrr_fast

nrows = 4000000

meta = pd.read_csv("_meta_feautres_all.csv", nrows=nrows)
print(meta.head())
meta["revrank"] = -meta["rank"]
print("rank", mrr_fast(meta, "revrank"))
for fn in glob.glob("*.csv"):
    if fn.startswith("_meta"):
        continue
    fs = pd.read_csv(fn, nrows=nrows)
    for col in fs:
        if col == "clickout_id":
            continue
        meta["test"] = fs[col] - meta["rank"]/100000
        meta["test_group"] = meta.groupby("clickout_id")["test"].rank("max", ascending=True)
        print(col, mrr_fast(meta, "test"), mrr_fast(meta, "test_group"))


