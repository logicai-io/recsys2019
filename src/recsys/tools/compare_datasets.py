import pandas as pd
import numpy as np

df_old = pd.read_csv("../../../data/events_sorted_trans_old.csv", nrows=10000)
df_new = pd.read_csv("../../../data/events_sorted_trans.csv", nrows=10000)

print(df_old.tail(10))

for col in df_old.columns:
    if col in df_new.columns:
        agree = np.mean(df_new[col].fillna(0).values == df_old[col].fillna(0).values)
        if agree < 1:
            print(col, np.mean(df_new[col].values == df_old[col].values))
            print(df_old[col].values[-20:])
            print(df_new[col].values[-20:])
    else:
        print(col, "missing")


print(df_old["last_price_diff"].value_counts())
print(df_new["last_price_diff"].value_counts())
