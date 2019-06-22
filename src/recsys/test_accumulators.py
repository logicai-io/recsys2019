import json

import pandas as pd
import numpy as np
from lightgbm import LGBMRanker, LGBMClassifier, LGBMRankerMRR2
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
    ItemCTR,
    ItemCTREMA,
    ItemAverageRank,
    ClickProbabilityClickOffsetTimeOffsetByDevice,
    UserItemAttentionSpan,
    SameImpressionsDifferentUser,
    SameFakeImpressionsDifferentUser,
    SameImpressionsDifferentUserTopN,
    ClickSequenceTrend,
    AccByKey,
    ItemCTRInteractions,
    ItemAttentionSpan, ItemCTRInSequence)
from recsys.data_generator.generate_training_data import FeatureGenerator
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast, mrr_fast_v2
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_3_no_eng
from tqdm import tqdm
from scipy import sparse as sp

accumulators = [
    PriceSorted(),
    ActionsTracker(),
    DistinctInteractions(name="clickout", action_types=["clickout item"]),
    DistinctInteractions(name="interact", action_types=ACTIONS_WITH_ITEM_REFERENCE),
    PairwiseCTR(),
    RankOfItemsFreshClickout(),
    GlobalClickoutTimestamp(),
    SequenceClickout(),
    ItemCTR(action_types=["clickout item"]),
    # AccByKey(ItemCTR(action_types=["clickout item"]), key="platform_device"),
    # AccByKey(ItemCTR(action_types=["clickout item"]), key="platform"),
    # AccByKey(ItemCTR(action_types=["clickout item"]), key="device"),
    ItemCTRInteractions(),
    # AccByKey(ItemCTRInteractions(), key="platform_device"),
    # AccByKey(ItemCTRInteractions(), key="platform"),
    # AccByKey(ItemCTRInteractions(), key="device"),
    RankBasedCTR(),
    # new
    AccByKey(RankBasedCTR(), key="platform_device"),
    AccByKey(RankBasedCTR(), key="platform"),
    AccByKey(RankBasedCTR(), key="device"),
    AccByKey(RankBasedCTR(simple=True), key="platform_device"),
    AccByKey(RankBasedCTR(simple=True), key="platform"),
    AccByKey(RankBasedCTR(simple=True), key="device"),
    # AccByKey(ItemAttentionSpan(), key="platform_device"),
    # AccByKey(ItemAttentionSpan(), key="platform"),
    # AccByKey(ItemAttentionSpan(), key="device"),
    # AccByKey(ItemCTRInSequence(), key="platform_device"),
    # AccByKey(ItemCTRInSequence(), key="platform"),
    # AccByKey(ItemCTRInSequence(), key="device"),
    # AccByKey(PairwiseCTR(), key="platform_device"),
    # AccByKey(PairwiseCTR(), key="platform"),
    # AccByKey(PairwiseCTR(), key="device"),
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

0.6375773044621943
By rank
1 0.6699690922510806
2 0.6213110794810293
3 0.6045843053441776
4 0.5941121201608742
5 0.5997246413817992
6 0.5585462886446919
7 0.571538007638577
8 0.546672009907304
9 0.6279860681646396
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


model = LGBMRankerMRR2(learning_rate=0.1, n_estimators=100, min_child_samples=5, min_child_weight=0.00001, n_jobs=-2)
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
