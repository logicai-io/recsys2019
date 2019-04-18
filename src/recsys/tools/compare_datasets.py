import pandas as pd
import numpy as np

df_new = pd.read_csv("../../../data/events_sorted_trans_new.csv")
df_old = pd.read_csv("../../../data/events_sorted_trans.csv")

print(df_old.tail(10))

for col in df_old.columns:
    if col in df_new.columns:
        agree = np.mean(df_new[col].values==df_old[col].values)
        if agree < 1:
            print(col, np.mean(df_new[col].values==df_old[col].values))
            print(df_old[col].values[-20:])
            print(df_new[col].values[-20:])
    else:
        print(col, "missing")

print("last_index_1")
print(df_new["last_index_1"].value_counts())
print("fake_last_index_1")
print(df_new["fake_last_index_1"].value_counts())
