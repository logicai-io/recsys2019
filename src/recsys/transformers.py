import json
import pathlib
from collections import Counter
from typing import List

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
    def __init__(self, ascending=False):
        self.ascending = ascending

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col != "clickout_id":
                X[col] = X.groupby("clickout_id")[col].rank("max", ascending=self.ascending) - 1
        X.drop("clickout_id", axis=1, inplace=True)
        return X


class LagNumericalFeaturesWithinGroup(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            if col != "clickout_id":
                for n in [1, 2]:
                    X[col + f"_shifted_p{n}_diff"] = X[col] - X.groupby(["clickout_id"])[col].shift(n).fillna(0)
                    X[col + f"_shifted_m{n}_diff"] = X[col] - X.groupby(["clickout_id"])[col].shift(-n).fillna(0)
                    X[col + f"_shifted_p{n}"] = X.groupby(["clickout_id"])[col].shift(n).fillna(0)
                    X[col + f"_shifted_m{n}"] = X.groupby(["clickout_id"])[col].shift(-n).fillna(0)
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


class WrapPandas(BaseEstimator, TransformerMixin):
    def __init__(self, trans):
        self.trans = trans

    def fit(self, *args):
        self.trans.fit(*args)
        return self

    def transform(self, X):
        X_ = self.trans.transform(X)
        return pd.DataFrame(X_, columns=self.trans.get_feature_names())


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


class ClickTrackerEng(BaseEstimator, TransformerMixin):

    def fit(self, X, *arg):
        return self

    @staticmethod
    def classify_sequence(seq):
        if len(seq) == 0:
            return "empty"
        elif len(set(seq)) == 1:
            return "constant"
        elif len(set(seq)) == len(seq) and min(seq) == seq[0] and max(seq) == seq[-1]:
            return "ideal sequence"
        elif seq == sorted(seq):
            return "non ideal sequence"
        elif len(set(seq)) == len(seq) and min(seq) == seq[-1] and max(seq) == seq[0]:
            return "ideal sequence rev"
        elif seq == sorted(seq, reverse=True):
            return "non ideal sequence rev"
        else:
            return "other"

    def transform(self, X):
        ranks = list(X["rank"].values)
        clickout_ids = list(X["clickout_id"].values)
        users_sessions: List[List[List[int]]] = [json.loads(el) for el in X['click_indices']]
        all_features = []
        for user_sessions, rank, clickout_id in zip(users_sessions, ranks, clickout_ids):
            user_features = self.extract_features_from_user_sessions(rank, user_sessions)
            user_features['clickout_id'] = clickout_id
            all_features.append(user_features)
        return all_features

    def extract_features_from_user_sessions(self, rank, user_sessions):
        user_features = {}
        classes = [self.classify_sequence(sess) for sess in user_sessions]
        if classes:
            last_session = user_sessions[-1]
            last_class = classes[-1]
            last_class_with_new_item = self.classify_sequence(last_session + [rank])
            if self.continuation(last_class, last_class_with_new_item):
                user_features['sequence_{}_continuation'.format(last_class_with_new_item)] = len(last_session)
            user_features['sequence_{}_len'.format(last_class)] = len(last_session)
        for cl, freq in Counter(classes).most_common():
            user_features['sequence_{}_freq'.format(cl)] = freq
        return user_features

    def continuation(self, last_class, last_class_with_new_item):
        """
        Check if there is a continuation of a sequence
        If a sequence is ideal and is continued then it is ok
        But also when the previous class is ideal then there is a possibility that the class will be not ideal
        but still sorted. The same is for constant -> non ideal sequence
        """
        if last_class_with_new_item == last_class:
            return True
        elif last_class_with_new_item.startswith("non ideal sequence") and last_class.startswith("ideal sequence"):
            return True
        elif last_class == "constant" and "sequence" in last_class_with_new_item:
            return True
        else:
            return False


if __name__ == '__main__':
    click = ClickTrackerEng()
    print(click.extract_features_from_user_sessions(3, [[3, 2, 1], [1, 2, 3], [0, 0, 0]]))
    print(click.extract_features_from_user_sessions(0, [[0, 0, 0]]))
    print(click.extract_features_from_user_sessions(3, [[1, 2, 3]]))
    print(click.extract_features_from_user_sessions(4, [[1, 2, 3]]))
    print(click.extract_features_from_user_sessions(4, [[0, 0, 0], [1, 1, 1], [1, 2, 3]]))
