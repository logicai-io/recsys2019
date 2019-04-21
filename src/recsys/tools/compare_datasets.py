import pandas as pd
import numpy as np

df_old = pd.read_csv("../../../data/events_sorted_trans_all_old.csv", nrows=100000)
df_new = pd.read_csv("../../../data/events_sorted_trans_all.csv", nrows=100000)

print(df_old.tail(10))

for col_old in df_old.columns:
    if col_old not in df_new.columns:
        print(col_old, "missing")
    found = False
    for col_new in df_new.columns:
        agree = np.mean(df_new[col_new].fillna(0).values == df_old[col_old].fillna(0).values)
        if agree == 1:
            found = True
            break
    if found == False:
        print(col_old, "Not found")
