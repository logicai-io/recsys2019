import json
import pathlib

import joblib
import numpy as np
from joblib import Parallel, delayed
from recsys.constants import COUNTRY_CODES
from recsys.metric import mrr_fast
from recsys.utils import jaccard, timer
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

Parallel, delayed
PATH_TO_IMM = pathlib.Path(__file__).parents[2] / "data" / "item_metadata_map.joblib"


class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self, path_to_imm=PATH_TO_IMM):
        self.imm = joblib.load(PATH_TO_IMM)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["country"] = X["city"].map(lambda x: x.split(",")[-1].strip())
        X["country_eq_platform"] = (X["country"].map(COUNTRY_CODES) == X["platform"]).astype(np.int)
        X["last_event_ts_dict"] = X["last_event_ts"].map(json.loads)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int)

        items_to_score = zip(X["item_id"], X["last_item_clickout"].fillna(0).map(int))
        with timer("calculating item similarity"):
            X["item_similarity_to_last_clicked_item"] = [jaccard(self.imm[a], self.imm[b]) if b != 0 else 0 for a, b in
                                                         tqdm(items_to_score, total=X.shape[0])]
        print("item_similarity_to_last_clicked_item mrr",
              mrr_fast(X.iloc[:100000], "item_similarity_to_last_clicked_item"))

        items_to_score = zip(X["user_item_interactions_list"].map(json.loads), X["item_id"])
        with timer("calculating avg item similarity"):
            X["avg_similarity_to_interacted_items"] = [
                np.mean([jaccard(self.imm[a], self.imm[b]) for a in items]) if items else 0 for items, b in
                tqdm(items_to_score)
            ]
        print("avg_similarity_to_interacted_items mrr", mrr_fast(X.iloc[:100000], "avg_similarity_to_interacted_items"))

        X["user_item_ctr"] = X["clickout_user_item_clicks"] / (
                X["clickout_user_item_impressions"] + 1
        )
        X["last_poi_item_ctr"] = X["last_poi_item_clicks"] / (
                X["last_poi_item_impressions"] + 1
        )
        # X["properties"] = X["item_id"].map(self.imm)
        # X["properties"].fillna("", inplace=True)
        # X["hour"] = X["timestamp"].map(lambda t: arrow.get(t).hour)
        X["is_rank_greater_than_prv_click"] = X["rank"] > X["last_item_index"]
        X["last_filter"].fillna("", inplace=True)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(
            np.int
        )
        return X


class RankFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col != "clickout_id":
                X[col] = X.groupby("clickout_id")[col].rank("max", ascending=False) - 1
        X.drop("clickout_id", axis=1, inplace=True)
        return X


class LagNumericalFeaturesWithinGroup(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col != "clickout_id":
                X[col + "_shifted_p1_diff"] = X[col] - X.groupby(["clickout_id"])[
                    col
                ].shift(1).fillna(0)
                X[col + "_shifted_m1_diff"] = X[col] - X.groupby(["clickout_id"])[
                    col
                ].shift(-1).fillna(0)
                X[col + "_shifted_p1"] = (
                    X.groupby(["clickout_id"])[col].shift(1).fillna(0)
                )
                X[col + "_shifted_m1"] = (
                    X.groupby(["clickout_id"])[col].shift(-1).fillna(0)
                )
        X.drop("clickout_id", axis=1, inplace=True)
        return X


class PandasToRecords(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        return X.to_dict(orient="records")


class PandasToNpArray(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        return X.values.astype(np.float)
