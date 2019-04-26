import warnings
from collections import defaultdict

import pandas as pd
from lightgbm import LGBMRanker, LGBMClassifier
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast, group_clickouts_into_list
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1, make_vectorizer_2
import numpy as np

warnings.filterwarnings("ignore")

df = pd.read_csv("../../data/events_sorted_trans_all.csv", nrows=1000000)

df_train, df_val = split_by_timestamp(df)

def select_pairs(clickouts, targets):
    negatives = defaultdict(list)
    positives = defaultdict(list)
    for i, (clickout_id, target) in enumerate(zip(clickouts, targets)):
        if target:
            positives[clickout_id].append(i)
        else:
            negatives[clickout_id].append(i)
    pairs_a = []
    pairs_b = []
    for key in negatives:
        n = len(negatives[key])
        if n == 1:
            continue
        if negatives[key] and positives[key]:
            for _ in range(n):
                a, b = np.random.choice(negatives[key], 2)
                pairs_a.append(a)
                pairs_b.append(b)
            for _ in range(n):
                a, b = np.random.choice(positives[key], 1)[0], np.random.choice(negatives[key], 1)[0]
                pairs_a.append(a)
                pairs_b.append(b)
                pairs_a.append(b)
                pairs_b.append(a)

    pairs_df = pd.DataFrame({'left': pairs_a, 'right': pairs_b}).drop_duplicates()
    return pairs_df

# vectorizer 1
vectorizer_1 = make_vectorizer_1()

mat_train = vectorizer_1.fit_transform(df_train, df_train["was_clicked"])
print(mat_train.shape)
mat_val = vectorizer_1.transform(df_val)
print(mat_val.shape)

model = LGBMRanker(n_estimators=1600)
model.fit(mat_train, df_train["was_clicked"], group=group_lengths(df_train["clickout_id"]))

df_train["click_proba"] = model.predict(mat_train)
df_val["click_proba"] = model.predict(mat_val)

print(mrr_fast(df_train, "click_proba"))
print(mrr_fast(df_val, "click_proba"))

# vectorizer_2 for pairwise model
# vectorizer_2 = make_vectorizer_2()
#
# mat_train_pairwise = vectorizer_2.fit_transform(df_train, df_train["was_clicked"])
# print(mat_train_pairwise.shape)
# mat_val_pairwise = vectorizer_2.transform(df_val)
# print(mat_val_pairwise.shape)
#
# df_pairs = select_pairs(df_train.clickout_id, df_train.was_clicked)
# pairs_a = df_pairs.left.values
# pairs_b = df_pairs.right.values
#
# mat_train_pairs = np.hstack([mat_train_pairwise[pairs_a, :], mat_train_pairwise[pairs_b, :], mat_train_pairwise[pairs_b, :] - mat_train_pairwise[pairs_a, :]])
# y_train = np.sign(df_train["was_clicked"].values[pairs_a] - df_train["was_clicked"].values[pairs_b])
#
# model = LGBMClassifier()
# model.fit(mat_train_pairs, y_train)
#
# select only 2 first predicted items and try to reverse their order
# items_probs = group_clickouts_into_list(df_val, "click_proba", append_index=True)
# pairs_a = []
# pairs_b = []
# for key in items_probs:
#     pairs_a.append(items_probs[key][0])
#     pairs_b.append(items_probs[key][1])
# pairs_a = np.array(pairs_a)
# pairs_b = np.array(pairs_b)
# mat_val_pairs = np.hstack([mat_val_pairwise[pairs_a, :], mat_val_pairwise[pairs_b, :], mat_val_pairwise[pairs_b, :] - mat_val_pairwise[pairs_a, :]])
# val_pairs_preds = model.predict_proba(mat_val_pairs)