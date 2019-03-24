import pandas as pd


old_df = pd.read_csv("../../data/events_sorted_trans.csv")
new_df = pd.read_csv("../../data/events_sorted_trans_test.csv")

for col in old_df.columns:
    if col in new_df.columns:
        print(col, (old_df[col] == new_df[col]).mean(), (new_df[col] == 0).mean())
