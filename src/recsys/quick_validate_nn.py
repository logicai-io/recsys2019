import os

from recsys.metric import mrr_fast
from recsys.nn import nn_fit_predict

os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing.pool import ThreadPool

import numpy as np

import warnings

import pandas as pd
from recsys.df_utils import split_by_timestamp
from recsys.vectorizers import make_vectorizer_2

warnings.filterwarnings("ignore")

df = pd.read_csv("../../data/events_sorted_trans_all.csv", nrows=2000000)

df_train, df_val = split_by_timestamp(df)

vectorizer = make_vectorizer_2()
mat_train = vectorizer.fit_transform(df_train, df_train["was_clicked"])
print(mat_train.shape)
mat_val = vectorizer.transform(df_val)
print(mat_val.shape)

with ThreadPool(processes=8) as pool:
    y_preds = pool.starmap(nn_fit_predict, [((mat_train, mat_val), df_train["was_clicked"].values)] * 8)

for n in range(1, len(y_preds)):
    df_val["nn_preds"] = np.vstack(y_preds[:n]).T.mean(axis=1)
    # df_val["nn_preds"] = y_preds[n-1]
    print(n, mrr_fast(df_val, "nn_preds"))

print("By rank")
for n in range(1, 10):
    print(n, mrr_fast(df_val[df_val["clickout_step_rev"] == n], "nn_preds"))

assert False

"""
0.6365213771646553
By rank
1 0.6691318996725455
2 0.6007534221519896
3 0.6185981922813633
4 0.5874910615366599
5 0.595400160091179
6 0.6117028935263851
7 0.5577631093186649
8 0.5881863593922418
9 0.5779483588307118
"""
