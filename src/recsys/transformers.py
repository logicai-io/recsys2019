from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEng(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["ctr"] = X["clickout_item_clicks"] / (X["clickout_item_impressions"] + 1)
        return X[["ctr"]]


class RankFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            X[col] = X.groupby("clickout_id")[col].rank("max", ascending=False) - 1
        X.drop("clickout_id", axis=1, inplace=True)
        return X