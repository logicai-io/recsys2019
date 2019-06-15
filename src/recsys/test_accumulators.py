import json

import pandas as pd
import numpy as np
from lightgbm import LGBMRanker, LGBMClassifier
from recsys.data_generator.accumulators import (
    ACTIONS_WITH_ITEM_REFERENCE,
    ActionsTracker,
    DistinctInteractions,
    PriceSorted,
    PairwiseCTR,
    RankOfItemsFreshClickout,
    ClickProbabilityClickOffsetTimeOffset,
    GlobalClickoutTimestamp,
    SequenceClickout,
    RankBasedCTR,
)
from recsys.data_generator.generate_training_data import FeatureGenerator
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast, mrr_fast_v2
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_3_no_eng
from tqdm import tqdm
from scipy import sparse as sp

"""
for col in ["fake_impressions_v2_user", "fake_impressions_v2_user_session", 
            "fake_impressions_v2_user_resets", "fake_impressions_v2_user_session_resets"]:
"""


accumulators = [
    PriceSorted(),
    ActionsTracker(),
    DistinctInteractions(name="clickout", action_types=["clickout item"]),
    DistinctInteractions(name="interact", action_types=ACTIONS_WITH_ITEM_REFERENCE),
    PairwiseCTR(),
    RankOfItemsFreshClickout(),
    GlobalClickoutTimestamp(),
    SequenceClickout(),
    RankBasedCTR()
    # ClickProbabilityClickOffsetTimeOffset(
    #     name="fake_clickout_prob_time_position_offset",
    #     action_types=ACTIONS_WITH_ITEM_REFERENCE,
    #     impressions_type="fake_impressions_raw",
    #     index_col="fake_index_interacted",
    #     probs_path="../../data/click_probs_by_index.joblib"
    # ),
    # ClickProbabilityClickOffsetTimeOffset(
    #     name="fake_clickout_prob_time_position_offset_v2_user",
    #     action_types=ACTIONS_WITH_ITEM_REFERENCE,
    #     impressions_type="fake_impressions_v2_user_raw",
    #     index_col="fake_impressions_v2_user_index",
    #     probs_path="../../data/click_probs_by_index.joblib"
    # ),
    # ClickProbabilityClickOffsetTimeOffset(
    #     name="fake_clickout_prob_time_position_offset_v2_user_session",
    #     action_types=ACTIONS_WITH_ITEM_REFERENCE,
    #     impressions_type="fake_impressions_v2_user_session_raw",
    #     index_col="fake_impressions_v2_user_session_index",
    #     probs_path="../../data/click_probs_by_index.joblib"
    # ),
    # ClickProbabilityClickOffsetTimeOffset(
    #     name="fake_clickout_prob_time_position_offset_v2_user_resets",
    #     action_types=ACTIONS_WITH_ITEM_REFERENCE,
    #     impressions_type="fake_impressions_v2_user_resets_raw",
    #     index_col="fake_impressions_v2_user_resets_index",
    #     probs_path="../../data/click_probs_by_index.joblib"
    # ),
    # ClickProbabilityClickOffsetTimeOffset(
    #     name="fake_clickout_prob_time_position_offset_v2_user_session_resets",
    #     action_types=ACTIONS_WITH_ITEM_REFERENCE,
    #     impressions_type="fake_impressions_v2_user_session_resets_raw",
    #     index_col="fake_impressions_v2_user_session_resets_index",
    #     probs_path="../../data/click_probs_by_index.joblib"
    # ),
]

"""
500k
0.6239103826449669
By rank
1 0.6633627256209672
2 0.5997924664937383
3 0.5711420157767378
4 0.5789740017498108
5 0.5392624885085371
6 0.5620807869887392
7 0.6063154068154069
8 0.556917981366362
9 0.5236013986013986


0.6272691141456231
By rank
1 0.6655270761353442
2 0.602912678571626
3 0.5723195447128363
4 0.5820829925710631
5 0.5596296676257162
6 0.5626214755254666
7 0.6316358136342608
8 0.5626260839680572
9 0.5318723175746586
"""

"""
0.6343673589319019
By rank
1 0.666705136967124
2 0.6199764010371607
3 0.5973204551587935
4 0.5839764046757063
5 0.5968199125460283
6 0.5641159715544564
7 0.5731059145366546
8 0.5312000449384351
9 0.6131196728885805
"""

csv = "../../data/events_sorted_trans_mini.csv"
feature_generator = FeatureGenerator(
    limit=1000000,
    accumulators=accumulators,
    save_only_features=False,
    input="../../data/events_sorted.csv",
    save_as=csv,
)
feature_generator.generate_features()

df = pd.read_csv(csv)
df["actions_tracker"] = df["actions_tracker"].map(json.loads)

features = [col for col in list(df.columns[27:]) if col != "actions_tracker"]
df_train, df_val = split_by_timestamp(df)
vectorizer = make_vectorizer_3_no_eng(numerical_features=features, numerical_features_for_ranking=features)

mat_train = vectorizer.fit_transform(df_train, df_train["was_clicked"])
print(mat_train.shape)
mat_val = vectorizer.transform(df_val)
print(mat_val.shape)


def mrr_metric(train_data, preds):
    mrr = mrr_fast_v2(train_data, preds, df_val["clickout_id"].values)
    return "error", mrr, True


model = LGBMRanker(learning_rate=0.1, n_estimators=100, min_child_samples=5, min_child_weight=0.00001, n_jobs=-2)
model.fit(
    mat_train,
    df_train["was_clicked"],
    group=group_lengths(df_train["clickout_id"]),
    # eval_set=[(mat_val, df_val["was_clicked"])],
    # eval_group=[group_lengths(df_val["clickout_id"])],
    # eval_metric=mrr_metric,
)

df_train["click_proba"] = model.predict(mat_train)
df_val["click_proba"] = model.predict(mat_val)

print(mrr_fast(df_val, "click_proba"))
print("By rank")
for n in range(1, 10):
    print(n, mrr_fast(df_val[df_val["clickout_step_rev"] == n], "click_proba"))

model = LGBMRanker(
    learning_rate=0.1,
    n_estimators=100,
    min_child_samples=5,
    min_child_weight=0.00001,
    # importance_type="gain",
    n_jobs=-2,
)
model.fit(
    df_train[features],
    df_train["was_clicked"],
    group=group_lengths(df_train["clickout_id"]),
    # eval_set=[(df_val[features], df_val["was_clicked"])],
    # eval_group=[group_lengths(df_val["clickout_id"])],
    # eval_metric=mrr_metric,
)

df_train["click_proba"] = model.predict(df_train[features])
df_val["click_proba"] = model.predict(df_val[features])
print(mrr_fast(df_val, "click_proba"))
for fname, imp in zip(features, model.feature_importances_):
    df_val[fname + "m"] = -df_val[fname]
    print(fname, imp, max(mrr_fast(df_val.sample(frac=1), fname + "m"), mrr_fast(df_val.sample(frac=1), fname)))


"""

def transpose_mat(mat_train, df_train):
    mat_train_t = np.zeros((df_train["clickout_id"].nunique(), mat_train.shape[1]*25), dtype=np.float32)
    mat_clickout_row = dict(zip(df_train["clickout_id"].unique(), np.arange(df_train["clickout_id"].nunique())))
    mat_clickout_y = dict(zip(df_train["clickout_id"], df_train["index_clicked"]))
    n_rows = mat_train.shape[0]
    n_cols = mat_train.shape[1]
    ranks = df_train["rank"].values
    clickouts = df_train["clickout_id"].values
    for i in tqdm(range(n_rows)):
        rank = ranks[i]
        i_t = mat_clickout_row[clickouts[i]]
        for j in range(n_cols):
            j_t = rank*n_cols + j
            mat_train_t[i_t, j_t] = mat_train[i, j]
    y_train_t = pd.Series(df_train["clickout_id"].unique()).map(mat_clickout_y)
    return sp.csr_matrix(mat_train_t), y_train_t


mat_train_t, y_train_t = transpose_mat(mat_train, df_train)
mat_val_t, y_val_t = transpose_mat(mat_val, df_val)

model = LGBMClassifier(learning_rate=0.1, n_estimators=100, min_child_samples=5, min_child_weight=0.00001, verbose=1, n_jobs=-2)
model.fit(
    sp.csr_matrix(mat_train_t),
    y_train_t,
    eval_set=[(sp.csr_matrix(mat_val_t), y_val_t)],
)

val_pred_proba = model.predict_proba(sp.csr_matrix(mat_val_t))[:,1:]
val_pred_proba_sort = np.argsort(-val_pred_proba, axis=1)
val_pred_proba_sort[np.arange(val_pred_proba.shape[0]), y_val_t]

assert False
"""
