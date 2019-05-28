import glob
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRankerMRR
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast, mrr_fast_v2
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings("ignore")

nrows = 2000000
df = pd.read_csv("../../data/events_sorted_trans_all.csv", nrows=nrows)

for fn in glob.glob("../../data/features/graph*.csv"): # + glob.glob("../../data/features/item*.csv") :
    new_df = pd.read_csv(fn, nrows=nrows)
    for col in new_df.columns:
        print(col)
        df[col] = new_df[col]

df_train, df_val = split_by_timestamp(df)

vectorizer = make_vectorizer_2()
mat_train = vectorizer.fit_transform(df_train, df_train["was_clicked"])
print(mat_train.shape)
mat_val = vectorizer.transform(df_val)
print(mat_val.shape)


def mrr_metric(train_data, preds):
    mrr = mrr_fast_v2(train_data, preds, df_val["clickout_id"].values)
    return "error", mrr, True

for alpha in [0.1,0.03,0.01]:
    model = SGDClassifier(loss="log", n_iter=1, alpha=alpha, shuffle=False)
    model.fit(mat_train, df_train["was_clicked"])

    df_train["click_proba"] = model.predict_proba(mat_train)[:,1]
    df_val["click_proba"] = model.predict_proba(mat_val)[:,1]

    print(mrr_fast(df_val, "click_proba"), mrr_fast(df_train, "click_proba"))
print("By rank")
for n in range(1, 10):
    print(n, mrr_fast(df_val[df_val["clickout_step_rev"] == n], "click_proba"))

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
