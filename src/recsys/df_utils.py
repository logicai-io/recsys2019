import numpy as np


def split_by_timestamp(df_all, col="timestamp", perc=90):
    split_timestamp = np.percentile(df_all.timestamp, perc)
    df_train = df_all[df_all[col] <= split_timestamp]
    df_val = df_all[(df_all[col] > split_timestamp)]
    return df_train, df_val
