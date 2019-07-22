import warnings

import pandas as pd
from lightgbm import LGBMRanker
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast, mrr_fast_v2
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1

warnings.filterwarnings("ignore")


def read_data():
    nrows = 200000
    df = pd.read_csv("../../data/events_sorted_trans_all.csv", nrows=nrows)
    df_train, df_val = split_by_timestamp(df)
    return df_train, df_val


df_train, df_val = read_data()
vectorizer = make_vectorizer_1()
mat_train = vectorizer.fit_transform(df_train, df_train["was_clicked"])
print(mat_train.shape)
mat_val = vectorizer.transform(df_val)
print(mat_val.shape)


def mrr_metric(train_data, preds):
    mrr = mrr_fast_v2(train_data, preds, df_val["clickout_id"].values)
    return "error", mrr, True


model = LGBMRanker(learning_rate=0.05, n_estimators=900, min_child_samples=5, min_child_weight=0.00001, n_jobs=-2)
model.fit(
    mat_train,
    df_train["was_clicked"],
    group=group_lengths(df_train["clickout_id"]),
    # sample_weight=np.where(df_train["clickout_step_rev"]==1,2,1),
    verbose=True,
    eval_set=[(mat_val, df_val["was_clicked"])],
    eval_group=[group_lengths(df_val["clickout_id"])],
    eval_metric=mrr_metric,
)

df_train["click_proba"] = model.predict(mat_train)
df_val["click_proba"] = model.predict(mat_val)

print(mrr_fast(df_val, "click_proba"))
print("By rank")
for n in range(1, 10):
    print(n, mrr_fast(df_val[df_val["clickout_step_rev"] == n], "click_proba"))
