# import joblib
import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
train["src"] = "train"
train["is_test"] = 0
test = pd.read_csv("test.csv")
test["src"] = "test"
test["is_test"] = (test["reference"].isnull() & (test["action_type"] == "clickout item")).astype(np.int)
events = pd.concat([train, test], axis=0)
events.sort_values(["timestamp", "user_id", "step"], inplace=True)
events.to_csv("events_sorted.csv", index=False)
