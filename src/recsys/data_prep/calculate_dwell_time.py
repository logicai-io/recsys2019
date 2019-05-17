# coding: utf-8

import numpy as np
import pandas as pd

train = pd.read_csv("../../../data/events_sorted.csv")
train["timestamp"] = pd.to_datetime(train["timestamp"], unit="s")
train = train.sort_values(["user_id", "session_id", "timestamp"]).reset_index(drop=True)
reference_first = np.zeros(len(train))
reference_first[0] = 1
check = train["reference"][0]

for i in range(1, len(train)):
    if check != train["reference"][i]:
        check = train["reference"][i]
        reference_first[i] = 1

dwell_df = pd.DataFrame(columns=["reference", "dwell_time"])
dwell_df["reference"] = train[reference_first == 1]["reference"]
dwell_df.reset_index(drop=True, inplace=True)

first_time = train[reference_first == 1]["timestamp"]
first_time.reset_index(drop=True, inplace=True)

last_time = pd.Series(np.roll(first_time, -1))

dwell_df["dwell_time"] = (last_time - first_time).astype("timedelta64[s]")

session_end = train[reference_first == 1].groupby(["user_id", "session_id"]).size().values
session_end = session_end.cumsum() - 1

dwell_df = dwell_df.drop(session_end)
dwell_df = dwell_df[dwell_df.reference.apply(lambda x: x.isnumeric())]

dwell_df_agg = dwell_df.groupby("reference").agg(["min", "max", "mean", "median"])
dwell_df_agg = dwell_df_agg.reset_index()
dwell_df_agg.columns = ["reference", "dwell_min", "dwell_max", "dwell_mean", "dwell_median"]
dwell_df_agg.to_csv("../../../data/dwell_time_agg.csv", index=False)
