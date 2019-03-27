import json
import pathlib

import arrow
import numpy as np
import pandas as pd
from recsys.constants import COUNTRY_CODES
from recsys.utils import reduce_mem_usage
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

PATH_TO_IMM = pathlib.Path().absolute().parents[1] / "data" / "item_metadata_map.joblib"
METADATA_DENSE = pathlib.Path().absolute().parents[1] / "data" / "item_metadata_dense.csv"


class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.metadata_dense = reduce_mem_usage(pd.read_csv(METADATA_DENSE).fillna(0))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["country"] = X["city"].map(lambda x: x.split(",")[-1].strip())
        X["country_eq_platform"] = (X["country"].map(COUNTRY_CODES) == X["platform"]).astype(np.int32)
        X["last_event_ts_dict"] = X["last_event_ts"].map(json.loads)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int32)
        X["user_item_ctr"] = X["clickout_user_item_clicks"] / (X["clickout_user_item_impressions"] + 1)
        X["last_poi_item_ctr"] = X["last_poi_item_clicks"] / (X["last_poi_item_impressions"] + 1)
        X["properties"] = X["item_id"].values
        X = pd.merge(X, self.metadata_dense, how="left", on="item_id")
        X["hour"] = X["timestamp"].map(lambda t: arrow.get(t).hour)
        X["is_rank_greater_than_prv_click"] = (X["rank"] > X["last_item_index"]).astype(np.int32)
        X["last_filter"].fillna("", inplace=True)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int32)
        for col in X.columns:
            if X[col].dtype == np.bool:
                X[col] = X[col].astype(np.int32)
        return X


class FeatureEngScala(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["country"] = X["city"].map(lambda x: x.split(",")[-1].strip())
        X["country_eq_platform"] = (X["country"].map(COUNTRY_CODES) == X["platform"]).astype(np.int32)
        for col in X.columns:
            if X[col].dtype == np.bool:
                X[col] = X[col].astype(np.int32)
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


class SelectNumerical(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        self.cols = [i for i in range(X.shape[1]) if X[:, i].dtype in [np.int, np.float]]
        return self

    def transform(self, X):
        return X[self.cols]


class ToCSR(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        return csr_matrix(X)


class PandasToNpArray(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        return X.values.astype(np.float)
