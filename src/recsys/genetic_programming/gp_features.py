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

# +
import pandas as pd
import numpy as np
import datatable as dt
from tqdm import tqdm_notebook
import graphviz
import json
from collections import defaultdict
from gplearn.functions import make_function
from sklearn.feature_extraction import DictVectorizer
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor, SymbolicClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ACTIONS_WITH_ITEM_REFERENCE = {
    "search for item",
    "interaction item info",
    "interaction item image",
    "interaction item deals",
    "interaction item rating",
    "clickout item",
}
# -

df = pd.read_csv("../../../data/events_sorted_trans_mini.csv", nrows=10000)

# price_cols = [col for col in df_all.columns if col.startswith("price_")]
# item_id_cols = [col for col in df_all.columns if col.startswith("item_id_impressions_clickout_item")]
vect = DictVectorizer(sparse=False)
X_at = vect.fit_transform(df["actions_tracker"].map(json.loads))
X = pd.DataFrame(
    np.hstack([df[["clickout_id", "rank"]].values, X_at]), columns=["clickout_id", "rank"] + vect.get_feature_names()
)
y = df["was_clicked"].values
print(list(X.shape))

X["clickout item_00_timestamp"].value_counts()


# +
def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)


logical = make_function(function=_logical, name="logical", arity=4)


def _eq(x1, x2):
    return np.where(x1 == x2, 1, 0)


eq = make_function(function=_eq, name="eq", arity=2)


est = SymbolicClassifier(
    parsimony_coefficient=0.001,
    random_state=1,
    population_size=100000,
    function_set=["add", "sub", "mul", "div", "min", "max", logical, eq],  # , 'sqrt', 'log', 'abs', 'neg', 'inv'],
    feature_names=X.columns,
    n_jobs=1,
    verbose=True,
)

X_tr, X_va, y_tr, y_va, df_tr, df_va = train_test_split(X, y, df)
est.fit(X_tr, y_tr)

va_pred = est.predict_proba(X_va)[:, 1]
tr_pred = est.predict_proba(X_tr)[:, 1]
# -

set(va_pred)

dot_data = est._program.export_graphviz()
# dot_data = [p for p in est.steps[0][1].transformer_list[0][1]._programs[0] if p][6].export_graphviz()
graph = graphviz.Source(dot_data)
graph

fimp = list(zip(X.columns, est.feature_importances_))
df_imp = pd.DataFrame.from_records(fimp, columns=["col", "imp"])
df_imp.sort_values("imp", ascending=False)

from recsys.metric import mrr_fast_v2, mrr_fast

for col in X.columns:
    print(col, X[col].sum(), mrr_fast_v2(df["was_clicked"], X[col] - 0.0001 * X["rank"], df["clickout_id"]))
