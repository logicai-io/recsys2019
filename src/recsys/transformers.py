import joblib
import numpy as np
import pandas as pd
from recsys.utils import jaccard
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEng(BaseEstimator, TransformerMixin):
    features = [
        "item_id",
        "last_item_clickout",
        "clickout_user_item_clicks",
        "clickout_user_item_impressions",
    ]

    def __init__(self):
        self.imm = joblib.load("../../data/item_metadata_map.joblib")
        self.item_metadata = pd.read_csv("../../data/item_metadata.csv")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["item_similarity_to_last_clicked_item"] = X.apply(
            lambda row: jaccard(
                self.imm[row["item_id"]], self.imm[row["last_item_clickout"]]
            ),
            axis=1,
        )
        X["user_item_ctr"] = X["clickout_user_item_clicks"] / (
            X["clickout_user_item_impressions"] + 1
        )
        X = pd.merge(X, self.item_metadata, on="item_id", how="left")
        X["properties"].fillna("", inplace=True)
        return X


class RankFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            X[col] = X.groupby("clickout_id")[col].rank("max", ascending=False) - 1
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
