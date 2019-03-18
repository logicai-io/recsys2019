import json
import pathlib
from multiprocessing.pool import ThreadPool

import arrow
import joblib
import numpy as np
import pandas as pd
from recsys.constants import COUNTRY_CODES
from recsys.utils import jaccard, reduce_mem_usage, timer
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


PATH_TO_IMM = pathlib.Path().absolute().parents[1] / "data" / "item_metadata_map.joblib"
METADATA_DENSE = pathlib.Path().absolute().parents[1] / "data" / "item_metadata_dense.csv"

class JaccardItemSim:
    def __init__(self):
        self.imm = joblib.load(PATH_TO_IMM)

    def list_to_item(self, other_items, item):
        if other_items:
            return np.mean([jaccard(self.imm[a], self.imm[item]) for a in other_items])
        else:
            return 0

    def two_items(self, a, b):
        if b != 0:
            return jaccard(self.imm[a], self.imm[b])
        else:
            return 0

    def list_to_item_star(self, v):
        other_items, item = v
        if other_items:
            return np.mean([jaccard(self.imm[a], self.imm[item]) for a in other_items])
        else:
            return 0

    def two_items_star(self, v):
        a, b = v
        if b != 0:
            return jaccard(self.imm[a], self.imm[b])
        else:
            return 0


class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self, path_to_imm=PATH_TO_IMM):
        self.jacc_sim = JaccardItemSim()
        self.metadata_dense = reduce_mem_usage(pd.read_csv(METADATA_DENSE).fillna(0))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["country"] = X["city"].map(lambda x: x.split(",")[-1].strip())
        X["country_eq_platform"] = (X["country"].map(COUNTRY_CODES) == X["platform"]).astype(np.int32)
        X["last_event_ts_dict"] = X["last_event_ts"].map(json.loads)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int32)

        jacc_sim = self.jacc_sim
        with timer("calculating item similarity"):
            with ThreadPool(8) as pool:
                items_to_score = zip(X["item_id"], X["last_item_clickout"].fillna(0).map(int))
                X["item_similarity_to_last_clicked_item"] = list(
                    tqdm(pool.imap(jacc_sim.two_items_star, items_to_score, chunksize=100))
                )

                items_to_score = zip(X["user_item_interactions_list"].map(json.loads), X["item_id"])
                X["avg_similarity_to_interacted_items"] = list(
                    tqdm(pool.imap(jacc_sim.list_to_item_star, items_to_score, chunksize=100))
                )

                items_to_score = zip(X["user_item_session_interactions_list"].map(json.loads), X["item_id"])
                X["avg_similarity_to_interacted_session_items"] = list(
                    tqdm(pool.imap(jacc_sim.list_to_item_star, items_to_score, chunksize=100))
                )

        X["user_item_ctr"] = X["clickout_user_item_clicks"] / (X["clickout_user_item_impressions"] + 1)
        X["last_poi_item_ctr"] = X["last_poi_item_clicks"] / (X["last_poi_item_impressions"] + 1)
        # X["properties"] = X["item_id"].map(self.imm)
        # X["properties"].fillna("", inplace=True)
        X = pd.merge(X, self.metadata_dense, how="left", on="item_id")

        X["hour"] = X["timestamp"].map(lambda t: arrow.get(t).hour)
        X["is_rank_greater_than_prv_click"] = X["rank"] > X["last_item_index"]
        X["last_filter"].fillna("", inplace=True)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int32)
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
                X[col + "_shifted_p1_diff"] = X[col] - X.groupby(["clickout_id"])[col].shift(1).fillna(0)
                X[col + "_shifted_m1_diff"] = X[col] - X.groupby(["clickout_id"])[col].shift(-1).fillna(0)
                X[col + "_shifted_p1"] = X.groupby(["clickout_id"])[col].shift(1).fillna(0)
                X[col + "_shifted_m1"] = X.groupby(["clickout_id"])[col].shift(-1).fillna(0)
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
