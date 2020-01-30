import pandas as pd

data_path = "../../../data/events_sorted_trans.csv"
df = pd.read_csv(data_path)
ftrl = pd.read_csv("ftrl_prob.csv")
df["ftrl_prob"] = ftrl["click"]
df.to_csv(data_path, index=False)
