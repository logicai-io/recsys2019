import warnings

import pandas as pd
from lightgbm import LGBMRanker
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1

warnings.filterwarnings("ignore")

df = pd.read_csv("../../data/events_sorted_trans_all.csv", nrows=1000000)

df_train, df_val = split_by_timestamp(df)
vectorizer = make_vectorizer_1()

mat_train = vectorizer.fit_transform(df_train, df_train["was_clicked"])
print(mat_train.shape)
mat_val = vectorizer.transform(df_val)
print(mat_val.shape)

model = LGBMRanker()
model.fit(mat_train, df_train["was_clicked"], group=group_lengths(df_train["clickout_id"]))

df_train["click_proba"] = model.predict(mat_train)
df_val["click_proba"] = model.predict(mat_val)

print(mrr_fast(df_train, "click_proba"))
print(mrr_fast(df_val, "click_proba"))
#
# scaler = make_pipeline(SanitizeSparseMatrix(), StandardScaler(with_mean=False))
# mat_train_s = scaler.fit_transform(mat_train, df_train["was_clicked"])
# mat_val_s = scaler.transform(mat_val)
#
# model_in = ks.Input(shape=(mat_train.shape[1],), dtype='float32', sparse=True)
# out = ks.layers.Dense(512, activation='relu')(model_in)
# out = ks.layers.Dense(256, activation='relu')(out)
# out = ks.layers.Dense(64, activation='relu')(out)
# out = ks.layers.Dense(1)(out)
# model = ks.Model(model_in, out)
# model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=0.001))
# model.fit(mat_train_s, df_train["was_clicked"], batch_size=256, epochs=3)
#
# df_train["click_proba_mlp"] = model.predict(mat_train_s)
# df_val["click_proba_mlp"] = model.predict(mat_val_s)
#
# print(mrr_fast(df_train, "click_proba_mlp"))
# print(mrr_fast(df_val, "click_proba_mlp"))
