# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# !pip install gplearn

from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
import pandas as pd
from lightgbm import LGBMRanker
from recsys.metric import mrr_fast
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
import numpy as np
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("../../../data/events_sorted_trans.csv")

numerical_features_py = ["rank", "last_index_1", "last_index_2", "last_index_3", "last_index_4", "last_index_5"]
df[numerical_features_py].fillna(-1000, inplace=True)
df["new_feature"] = 0.0

df_gp_train = df[:200000]
df_gp_test = df[200000:300000]
X_train = df_gp_train[numerical_features_py]
y_train = df_gp_train["was_clicked"]
X_test = df_gp_test[numerical_features_py]
y_test = df_gp_test["was_clicked"]

tree = LGBMClassifier()  # class_weight="balanced"))
tree.fit(X_train, y_train)
df_gp_test["new_feature_lgbm"] = tree.predict_proba(X_test)[:, 1]
print(mrr_fast(df_gp_test, "new_feature_lgbm"))
df_gp_train["new_feature_lgbm"] = tree.predict_proba(X_train)[:, 1]
print(mrr_fast(df_gp_train, "new_feature_lgbm"))


def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)


logical = make_function(function=_logical, name="logical", arity=4)

function_set = ["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min", logical]
gp = SymbolicTransformer(
    generations=20,
    population_size=1000,
    hall_of_fame=100,
    n_components=10,
    function_set=function_set,
    parsimony_coefficient=0.0003,
    max_samples=0.9,
    verbose=1,
    random_state=0,
    n_jobs=1,
)

gp.fit(X_train, y_train)

for p in gp._best_programs:
    print(p)

X_train_gp = gp.transform(X_train)
X_test_gp = gp.transform(X_test)

tree = LGBMClassifier()  # class_weight="balanced"))
tree.fit(np.hstack([X_train, X_train_gp]), y_train)
df_gp_test["new_feature_lgbm"] = tree.predict_proba(np.hstack([X_test, X_test_gp]))[:, 1]
print(mrr_fast(df_gp_test, "new_feature_lgbm"))
df_gp_train["new_feature_lgbm"] = tree.predict_proba(np.hstack([X_train, X_train_gp]))[:, 1]
print(mrr_fast(df_gp_train, "new_feature_lgbm"))

for col in range(X_train_gp.shape[1]):
    df_gp_test["click_proba"] = X_test_gp[:, col]
    mrr = mrr_fast(df_gp_test, "click_proba")
    print(col, mrr)

for i, col in enumerate(numerical_features_py):
    df_gp_test["click_proba"] = df_gp_test[col].fillna(df_gp_test[col].unique()[0])
    mrr = mrr_fast(df_gp_test, "click_proba")
    print(i, col, mrr)

df_gp_test_ = df_gp_test.sort_values("current_filters")

df_gp_test_["current_filters"].value_counts()
