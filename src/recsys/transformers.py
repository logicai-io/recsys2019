import json
import pathlib
from copy import deepcopy

import arrow
import joblib
import numpy as np
import pandas as pd
from recsys.constants import COUNTRY_CODES
from recsys.timestamp import convert_dt_to_timezone
from recsys.utils import reduce_mem_usage
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

PATH_TO_IMM = pathlib.Path().absolute().parents[1] / "data" / "item_metadata_map.joblib"
METADATA_DENSE = pathlib.Path().absolute().parents[1] / "data" / "item_metadata_dense.csv"
PRICE_PCT_PER_CITY = pathlib.Path().absolute().parents[1] / "data" / "price_pct_by_city.joblib"
PRICE_PCT_PER_PLATFORM = pathlib.Path().absolute().parents[1] / "data" / "price_pct_by_platform.joblib"
PRICE_RANK_PER_ITEM = pathlib.Path().absolute().parents[1] / "data" / "item_prices_rank.joblib"
LSTM_USER_SESSION = pathlib.Path().absolute().parents[1] / "data" / "lstm" / "oof_predictions_user_session.csv"
LSTM_USER = pathlib.Path().absolute().parents[1] / "data" / "lstm" / "oof_predictions_user.csv"


class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imm = joblib.load(PATH_TO_IMM)
        metadata_dense = reduce_mem_usage(pd.read_csv(METADATA_DENSE).fillna(0))
        X["country"] = X["city"].map(lambda x: x.split(",")[-1].strip())
        X["item_id_cat"] = X["item_id"].map(str)
        X["country_eq_platform"] = (X["country"].map(COUNTRY_CODES) == X["platform"]).astype(np.int32)
        X["last_event_ts_dict"] = X["last_event_ts"].map(json.loads)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int32)
        X["last_poi_item_ctr"] = X["last_poi_item_clicks"] / (X["last_poi_item_impressions"] + 1)
        X["user_item_ctr"] = X["clickout_user_item_clicks"] / (X["clickout_user_item_impressions"] + 1)
        X["properties"] = [str(x) for x in X["item_id"].map(imm)]
        X = pd.merge(X, metadata_dense, how="left", on="item_id")
        X["hour"] = X["timestamp"].map(lambda t: arrow.get(t).hour)
        X["is_rank_greater_than_prv_click"] = (X["rank"] > X["last_item_index"]).astype(np.int32)
        X["last_filter"].fillna("", inplace=True)
        X["clicked_before"] = (X["item_id"] == X["last_item_clickout"]).astype(np.int32)
        X["last_poi"].fillna("", inplace=True)
        X["alltime_filters"].fillna("", inplace=True)
        X["user_id_1cat"] = X["user_id"].map(lambda x: x[0])

        # add price per city percentile
        price_pct_by_city = joblib.load(PRICE_PCT_PER_CITY)
        keys = list(zip(X["city"], X["price"]))
        X["price_pct_by_city"] = [price_pct_by_city[k] for k in keys]

        price_pct_by_city = joblib.load(PRICE_PCT_PER_PLATFORM)
        keys = list(zip(X["platform"], X["price"]))
        X["price_pct_by_platform"] = [price_pct_by_city[k] for k in keys]

        X["datetime"] = X["timestamp"].map(arrow.get)
        X["datetime_local"] = X.apply(convert_dt_to_timezone, axis=1)
        X["datetime_hour"] = X["datetime"].map(lambda x: x.hour)
        X["datetime_minute"] = X["datetime"].map(lambda x: x.minute)
        X["datetime_local_hour"] = X["datetime"].map(lambda x: x.hour)
        X["datetime_local_minute"] = X["datetime"].map(lambda x: x.minute)
        X.drop(["datetime", "datetime_local"], axis=1, inplace=True)

        price_rank = joblib.load(PRICE_RANK_PER_ITEM)
        X = pd.merge(X, price_rank, how="left", on=["item_id", "price"])

        lstm_user_session = pd.read_csv(LSTM_USER_SESSION).rename(columns={'prob': 'lstm_user_session_prob'})
        X = pd.merge(X, lstm_user_session, how="left", on=["user_id", "session_id", "step"])

        lstm_user = pd.read_csv(LSTM_USER).rename(columns={'prob': 'lstm_user_prob'})
        X = pd.merge(X, lstm_user, how="left", on=["user_id", "session_id", "step"])

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
    def __init__(self, drop_clickout_id=True, ascending=False):
        self.drop_clickout_id = drop_clickout_id
        self.ascending = ascending

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col != "clickout_id":
                X[col] = X.groupby("clickout_id")[col].rank("max", ascending=self.ascending) - 1
        if self.drop_clickout_id:
            X.drop("clickout_id", axis=1, inplace=True)
        return X


class NormalizeRanking(BaseEstimator, TransformerMixin):
    def __init__(self, drop_clickout_id=True):
        self.drop_clickout_id = drop_clickout_id

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col != "clickout_id":
                X[col] = X.groupby("clickout_id")[col].fillna(0).transform(lambda x: (x - x.mean()) / (x.std() + 1))
        if self.drop_clickout_id:
            X.drop("clickout_id", axis=1, inplace=True)
        return X


class LagNumericalFeaturesWithinGroup(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1, drop_clickout_id=True, groupby="clickout_id"):
        self.offset = offset
        self.drop_clickout_id = drop_clickout_id
        self.groupby = groupby

    def fit(self, X, y=None):
        self.diff_cols = []
        for col in X.columns:
            if col != "clickout_id":
                nunique = X.groupby(self.groupby)[col].agg(lambda x: len(set(x)))
                if np.any(nunique != 1):
                    self.diff_cols.append(col)
        return self

    def transform(self, X):
        new_cols = []
        for col in self.diff_cols:
            X[col + "_shifted_p1_diff"] = X[col] - X.groupby([self.groupby])[col].shift(self.offset).fillna(0)
            new_cols.append(col + "_shifted_p1_diff")
            X[col + "_shifted_m1_diff"] = X[col] - X.groupby([self.groupby])[col].shift(-self.offset).fillna(0)
            new_cols.append(col + "_shifted_m1_diff")
            X[col + "_shifted_p1"] = X.groupby([self.groupby])[col].shift(self.offset).fillna(0)
            new_cols.append(col + "_shifted_p1")
            X[col + "_shifted_m1"] = X.groupby([self.groupby])[col].shift(-self.offset).fillna(0)
            new_cols.append(col + "_shifted_m1")
        if self.drop_clickout_id:
            X.drop("clickout_id", axis=1, inplace=True)
        return X[new_cols]


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
        return self

    def transform(self, X):
        X.data[np.isnan(X.data)] = 0
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


class SparsityFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_nnz=None):
        self.min_nnz = min_nnz

    def fit(self, X, y=None):
        self.sparsity = X.getnnz(0)
        return self

    def transform(self, X):
        return X[:, self.sparsity >= self.min_nnz]


class FeaturesAtAbsoluteRank(BaseEstimator, TransformerMixin):
    def __init__(self, rank=1, normalize=False):
        self.rank = rank
        self.normalize = normalize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        suffix = "_rank_{}".format(self.rank)
        X_ranked = X[X["rank"] == self.rank]
        X_all = pd.merge(X, X_ranked, how="left", on="clickout_id", suffixes=("", suffix))
        cols = [c for c in X_all.columns if c.endswith(suffix) and c not in ("rank" + suffix, "clickout_id" + suffix)]
        orig_cols = [c.replace(suffix, "") for c in X.columns if c not in ("rank", "clickout_id")]
        # import ipdb; ipdb.set_trace()
        if self.normalize:
            return (X_all[cols].fillna(0).values - X_all[orig_cols].fillna(0).values).astype(np.float32)
        else:
            return X_all[cols].fillna(0).astype(np.float32)


class DivideByRanking(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        for col in X.columns:
            if col == "rank":
                continue
            X[col] = X[col] / (X["rank"] + 1)
        return X
