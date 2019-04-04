import warnings

import pandas as pd
from lightgbm import LGBMRanker, LGBMClassifier
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast
from recsys.transformers import SanitizeSparseMatrix
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1, numerical_features_py, numerical_features_for_ranking_py
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

df = pd.read_csv("../../data/events_sorted_trans.csv", nrows=1000000)
# col = "avg_price_similarity_to_interacted_session_items"
# df[col + "_rank"] = df.groupby("clickout_id")[col].rank("max", ascending=False)
# df[col + "m"] = -df["avg_price_similarity_to_interacted_session_items"]
# print(mrr_fast(df, "avg_price_similarity_to_interacted_session_items"))
# print(mrr_fast(df, "avg_price_similarity_to_interacted_session_itemsm"))
# print(mrr_fast(df, "avg_price_similarity_to_interacted_session_items_rank"))
# print(mrr_fast(df, "user_rank_preference_rank"))
# print(df["user_rank_dict"].unique())
# assert False

df_train, df_val = split_by_timestamp(df)
vectorizer = make_vectorizer_1()

mat_train = vectorizer.fit_transform(df_train)
mat_val = vectorizer.transform(df_val)

model = LGBMRanker()
model.fit(mat_train, df_train["was_clicked"], group=group_lengths(df_train["clickout_id"]))

df_train["click_proba"] = model.predict(mat_train)
df_val["click_proba"] = model.predict(mat_val)

print(mrr_fast(df_train, "click_proba"))
print(mrr_fast(df_val, "click_proba"))

model = make_pipeline(SanitizeSparseMatrix(), StandardScaler(with_mean=False), SGDClassifier(loss="log"))
model.fit(mat_train, df_train["was_clicked"])

df_train["click_proba_sgd"] = model.predict_proba(mat_train)[:, 1]
df_val["click_proba_sgd"] = model.predict_proba(mat_val)[:, 1]

print(mrr_fast(df_train, "click_proba_sgd"))
print(mrr_fast(df_val, "click_proba_sgd"))

assert False

vectorizer = make_vectorizer_1(
    numerical_features=numerical_features_py + ["click_proba_sgd"],
    numerical_features_for_ranking=numerical_features_for_ranking_py + ["click_proba_sgd"],
)

mat_train2 = vectorizer.fit_transform(df_train)
mat_val2 = vectorizer.transform(df_val)

model = LGBMRanker()
model.fit(mat_train2, df_train["was_clicked"], group=group_lengths(df_train["clickout_id"]))

df_train["click_proba2"] = model.predict(mat_train2)
df_val["click_proba2"] = model.predict(mat_val2)
print(mrr_fast(df_train, "click_proba2"))
print(mrr_fast(df_val, "click_proba2"))

vectorizer = make_vectorizer_1(
    numerical_features=numerical_features_py + ["click_proba", "click_proba2"],
    numerical_features_for_ranking=numerical_features_for_ranking_py + ["click_proba", "click_proba2"],
)

mat_train3 = vectorizer.fit_transform(df_train)
mat_val3 = vectorizer.transform(df_val)

model = LGBMRanker()
model.fit(mat_train3, df_train["was_clicked"], group=group_lengths(df_train["clickout_id"]))

df_train["click_proba3"] = model.predict(mat_train3)
df_val["click_proba3"] = model.predict(mat_val3)
print(mrr_fast(df_train, "click_proba3"))
print(mrr_fast(df_val, "click_proba3"))

# before tr 0.6340 va 0.6187
# after va 6233
