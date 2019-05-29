from collections import defaultdict

import arrow
import numpy as np
import pandas as pd

train = pd.read_csv("../../../data/train.csv")
train["src"] = "train"
train["is_test"] = 0

test = pd.read_csv("../../../data/test.csv")
test["src"] = "test"
test["is_test"] = (test["reference"].isnull() & (test["action_type"] == "clickout item")).astype(np.int)

assert np.all(train.columns == test.columns)

events = pd.concat([train, test], axis=0)
events.sort_values(["timestamp", "user_id", "step"], inplace=True)
events["fake_impressions"] = events.groupby(["user_id", "session_id"])["impressions"].bfill()
events["fake_prices"] = events.groupby(["user_id", "session_id"])["prices"].bfill()

events["clickout_step_rev"] = (
    events.groupby(["action_type", "session_id"])["step"].rank("max", ascending=False).astype(np.int)
)
events["clickout_step"] = (
    events.groupby(["action_type", "session_id"])["step"].rank("max", ascending=True).astype(np.int)
)
events["clickout_max_step"] = events["clickout_step"] + events["clickout_step_rev"] - 1

# select the last day for validation
events["dt"] = events["timestamp"].map(lambda x: str(arrow.get(x).date()))
events_val = events[(events["dt"] == "2018-11-06") & (events["action_type"] == "clickout item")]
events_val["user_clickout_step_rev"] = (
    events_val.groupby(["action_type", "user_id"])["step"].rank("max", ascending=False).astype(np.int)
)
val_last_clickouts = events_val[events_val["user_clickout_step_rev"] == 1][["user_id", "session_id", "step"]]
val_last_clickouts["is_val"] = 1
events = pd.merge(events, val_last_clickouts, on=["user_id", "session_id", "step"], how="left")
events["is_val"].fillna(0, inplace=True)
events["is_val"] = events["is_val"].astype(np.int)
events.to_csv("../../../data/events_sorted.csv", index=False)
