import glob
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRankerMRR
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast, mrr_fast_v2
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2

warnings.filterwarnings("ignore")

nrows = 2000000
df = pd.read_csv("../../data/events_sorted_trans_all.csv", nrows=nrows)

for fn in glob.glob("../../data/features/graph*.csv") + glob.glob("../../data/features/sgd*.csv"):
    new_df = pd.read_csv(fn, nrows=nrows)
    for col in new_df.columns:
        print(col)
        df[col] = new_df[col]

df_train, df_val = split_by_timestamp(df)

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

"""
0.6379943651575                                                                                          
By rank                                                                                                  
1 0.6706915117507509                                                                                     
2 0.6007690434139217                                                                                     
3 0.6259768731511526                                                                                     
4 0.5845607369498449                                                                                     
5 0.6017779624490083                                                                                     
6 0.6133978043066618
7 0.5658811435478103
8 0.5753191571500396
9 0.5907524859205532

[50]    valid_0's ndcg@1: 0.490718      valid_0's error: 0.619293
[100]   valid_0's ndcg@1: 0.494754      valid_0's error: 0.624176
[200]   valid_0's ndcg@1: 0.502018      valid_0's error: 0.630874
[300]   valid_0's ndcg@1: 0.505016      valid_0's error: 0.633441
[400]   valid_0's ndcg@1: 0.506053      valid_0's error: 0.63444
[500]   valid_0's ndcg@1: 0.506745      valid_0's error: 0.635387
[600]   valid_0's ndcg@1: 0.507091      valid_0's error: 0.635656
[700]   valid_0's ndcg@1: 0.508475      valid_0's error: 0.636692
[800]   valid_0's ndcg@1: 0.50859       valid_0's error: 0.636998
[900]   valid_0's ndcg@1: 0.510435      valid_0's error: 0.637857
"""
