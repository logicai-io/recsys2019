import json
import pathlib
from copy import deepcopy

import arrow
import joblib
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
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imm = joblib.load(PATH_TO_IMM)
        metadata_dense = reduce_mem_usage(pd.read_csv(METADATA_DENSE).fillna(0))
        X["country"] = X["city"].map(lambda x: x.split(",")[-1].strip())
        X["country_eq_platform"] = (X["country"].map(COUNTRY_CODES) == X["platform"]).astype(np.int32)
        X["last_event_ts_dict"] = X["last_event_ts"].map(json.loads)
        # X["user_rank_dict"] = X["user_rank_dict"].map(json.loads)
        # X["user_session_rank_dict"] = X["user_session_rank_dict"].map(json.loads)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int32)
        X["user_item_ctr"] = X["clickout_user_item_clicks"] / (X["clickout_user_item_impressions"] + 1)
        X["last_poi_item_ctr"] = X["last_poi_item_clicks"] / (X["last_poi_item_impressions"] + 1)
        X["properties"] = [str(x) for x in X["item_id"].map(imm)]
        X = pd.merge(X, metadata_dense, how="left", on="item_id")
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
    def __init__(self, drop_clickout_id=True):
        self.drop_clickout_id = drop_clickout_id

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col != "clickout_id":
                X[col] = X.groupby("clickout_id")[col].rank("max", ascending=False) - 1
        if self.drop_clickout_id:
            X.drop("clickout_id", axis=1, inplace=True)
        return X


class LagNumericalFeaturesWithinGroup(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1, drop_clickout_id=True, groupby="clickout_id"):
        self.offset = offset
        self.drop_clickout_id = drop_clickout_id
        self.groupby = groupby

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col != "clickout_id":
                X[col + "_shifted_p1_diff"] = X[col] - X.groupby([self.groupby])[col].shift(self.offset).fillna(0)
                X[col + "_shifted_m1_diff"] = X[col] - X.groupby([self.groupby])[col].shift(-self.offset).fillna(0)
                X[col + "_shifted_p1"] = X.groupby([self.groupby])[col].shift(self.offset).fillna(0)
                X[col + "_shifted_m1"] = X.groupby([self.groupby])[col].shift(-self.offset).fillna(0)
        if self.drop_clickout_id:
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


class MinimizeNNZ(BaseEstimator, TransformerMixin):
    """
    Offset the values so that the most frequent is offset to 0 if it is the most common one
    """

    def fit(self, X, *args):
        self.offsets = []
        for col in X.columns:
            v = X[col]
            dom = v.value_counts().index[0]
            vmin = v.min()
            vmax = v.max()
            if dom == vmax or dom == vmin:
                self.offsets.append(-dom)
            else:
                self.offsets.append(0)
        return self

    def transform(self, X):
        for col, offset in zip(X.columns, self.offsets):
            if offset != 0:
                X[col] += offset
        return X


class SanitizeSparseMatrix(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self.datamax = np.nanmax(X.data)
        return self

    def transform(self, X):
        X.data[np.isnan(X.data)] = 0
        X.data = X.data.clip(0, self.datamax)
        return X


class RemoveDuplicatedColumnsDF(BaseEstimator, TransformerMixin):
    def fit(self, X):
        groups = X.columns.to_series().groupby(X.dtypes).groups
        self.duplicate_cols = []
        for t, v in groups.items():
            cs = X[v].columns
            vs = X[v]
            lcs = len(cs)
            for i in range(lcs):
                ia = vs.iloc[:, i].values
                for j in range(i + 1, lcs):
                    ja = vs.iloc[:, j].values
                    if np.array_equiv(ia, ja):
                        self.duplicate_cols.append(cs[i])
                        break
        return self

    def transform(self, X):
        X = X.drop(self.duplicate_cols, axis=1)
        return X


class NormalizeClickSequence(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        X["click_index_sequence_dec"] = X["click_index_sequence"].map(json.loads)
        X["click_index_sequence_norm"] = X.apply(self.normalize_seq, axis=1)
        X["click_index_sequence_text"] = X["click_index_sequence_norm"].map(self.encode_as_text)
        return X

    def normalize_seq(self, row):
        seq = deepcopy(row["click_index_sequence_dec"])
        # return [[ind-row["rank"] if ind else 'X' for ind in session if ind] for session in seq]
        return [[ind if ind else "X" for ind in session if ind] for session in seq]

    def encode_as_text(self, seq):
        return "BEG " + " , ".join([" ".join(["N" + str(ind) for ind in session]) for session in seq]) + " END"


if __name__ == "__main__":
    df = pd.DataFrame({"click_index_sequence": ["[[0]]"] * 4 + ["[[1,2,3],[0]]"] * 2, "rank": [0, 1, 2, 3, 0, 1]})
    print(NormalizeClickSequence().fit_transform(df))
