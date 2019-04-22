import hashlib

import numpy as np
import pandas as pd


def hash_str(text):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def partition_row(row):
    hashn = hash_str(row["user_id"])
    if (row["is_val"] == 0) and (row["is_test"] == 0):
        find = hashn % 25
        return "01_train_%04d" % find
    elif (row["is_val"] == 1) and (row["is_test"] == 0):
        return "02_val_%04d" % 0
    elif row["is_test"] == 1:
        find = hashn % 4
        return "03_test_%04d" % find
    else:
        raise ValueError("No matching value to split row")


train = pd.read_csv("../../../data/train.csv")
train["src"] = "train"
train["is_test"] = 0
train["is_val"] = 0

test = pd.read_csv("../../../data/test.csv")
validation_set = pd.read_csv("../../../data/validation_items.csv").drop("clickout_id", axis=1)
validation_set["is_val"] = 1
validation_set["is_val"] = validation_set["is_val"].astype(np.int)
test["src"] = "test"
test["is_test"] = (test["reference"].isnull() & (test["action_type"] == "clickout item")).astype(np.int)
test = pd.merge(test, validation_set, how="left", on=["user_id", "session_id", "timestamp", "step"])
test["is_val"].fillna(0, inplace=True)

assert np.all(train.columns == test.columns)

events: pd.DataFrame = pd.concat([train, test], axis=0)
events["is_val"] = events["is_val"].astype(np.int)
events.sort_values(["timestamp", "user_id", "step"], inplace=True)
events["fake_impressions"] = events.groupby(["user_id", "session_id"])["impressions"].bfill()
events["fake_prices"] = events.groupby(["user_id", "session_id"])["prices"].bfill()
events.insert(0, "partition_id", events.apply(partition_row, axis=1))
events.to_csv("../../../data/events_sorted.csv", index=False)
