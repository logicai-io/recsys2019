import warnings

import pandas as pd
from lightgbm import LGBMRanker
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1

warnings.filterwarnings("ignore")

df = pd.read_csv("../../data/events_sorted_trans.csv", nrows=1000000)

df_train, df_val = split_by_timestamp(df)
vectorizer = make_vectorizer_1()

mat_train = vectorizer.fit_transform(df_train)
print(mat_train.shape)
mat_val = vectorizer.transform(df_val)
print(mat_val.shape)

model = LGBMRanker()
model.fit(mat_train, df_train["was_clicked"], group=group_lengths(df_train["clickout_id"]))

df_train["click_proba"] = model.predict(mat_train)
df_val["click_proba"] = model.predict(mat_val)

print(mrr_fast(df_train, "click_proba"))
print(mrr_fast(df_val, "click_proba"))
