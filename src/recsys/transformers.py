import joblib
import numpy as np
import pandas as pd
from recsys.utils import jaccard, timer
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imm = joblib.load("../../data/item_metadata_map.joblib")
        self.item_metadata = pd.read_csv("../../data/item_metadata.csv")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        items_to_score = list(zip(X["item_id"], X["last_item_clickout"]))
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int)
        with timer("calculating item similarity"):
            X["item_similarity_to_last_clicked_item"] = [
                jaccard(self.imm[a], self.imm[b]) for a, b in tqdm(items_to_score)
            ]
        X["user_item_ctr"] = X["clickout_user_item_clicks"] / (
            X["clickout_user_item_impressions"] + 1
        )
        X["last_poi_item_ctr"] = X["last_poi_item_clicks"] / (
            X["last_poi_item_impressions"] + 1
        )
        X = pd.merge(X, self.item_metadata, on="item_id", how="left")
        X["properties"].fillna("", inplace=True)
        # X["hour"] = X["timestamp"].map(lambda t: arrow.get(t).hour)
        X["is_rank_greater_than_prv_click"] = X["rank"] > X["last_item_index"]
        X["last_filter"].fillna("", inplace=True)
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
