import pandas as pd
from lightgbm import LGBMRanker
from recsys.data_generator.accumulators import PriceSorted, ItemIDS
from recsys.data_generator.generate_training_data import FeatureGenerator
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast_v2, mrr_fast
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_3, make_vectorizer_3_no_eng

accumulators = [PriceSorted()]

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
# oof_predictions = pd.read_csv("../../data/lstm/oof_predictions_user.csv")
# df = pd.merge(df, oof_predictions, how="left", on=["user_id", "session_id", "item_id", "step"])

features = list(df.columns[22:])

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

model = LGBMRanker(learning_rate=0.1,
                   n_estimators=100,
                   min_child_samples=5,
                   min_child_weight=0.00001,
                   # importance_type="gain",
                   n_jobs=-2)
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
    print(fname, imp)
