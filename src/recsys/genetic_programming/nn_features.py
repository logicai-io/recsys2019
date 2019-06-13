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
from collections import defaultdict
from gplearn.functions import make_function

ACTIONS_WITH_ITEM_REFERENCE = {
    "search for item",
    "interaction item info",
    "interaction item image",
    "interaction item deals",
    "interaction item rating",
    "clickout item",
}
# -

df = pd.read_csv("../../../data/events_sorted.csv", nrows=4_000_000)

all_obs = []
for user_id, df_user in tqdm_notebook(df.iloc[:4_000_000].groupby("user_id"), mininterval=1):
    obs = {}
    events_dict = defaultdict(list)
    events_interaction = []
    max_timestamp = 0

    # aggregate to lists
    for row in df_user.to_dict(orient="records"):
        obs["mobile"] = int(row["device"] == "mobile")
        action_type = row["action_type"].replace(" ", "_")
        events_dict[action_type].append(row)
        if row["action_type"] in ACTIONS_WITH_ITEM_REFERENCE:
            events_interaction.append(row)
        max_timestamp = max(max_timestamp, row["timestamp"])

    for action_type in events_dict.keys():
        for event_num, row in enumerate(events_dict[action_type][::-1]):
            if event_num == 0 and action_type == "clickout_item":
                impressions = row["impressions"].split("|")
                prices = row["prices"].split("|")
                obs["item_count"] = len(impressions)
                if row["reference"] in impressions:
                    obs["index_clicked"] = impressions.index(row["reference"])

            if event_num <= 10:
                if action_type == "clickout_item":
                    for rank, (item_id, price) in enumerate(zip(impressions, prices)):
                        price = int(price)
                        item_id = int(item_id)
                        obs[f"co_item_id_{rank:02d}_{event_num:02d}"] = np.log1p(item_id)
                        obs[f"co_price_{rank:02d}_{event_num:02d}"] = np.log1p(price)

                obs[f"{action_type}_{event_num:02d}_timestamp"] = np.log1p(max_timestamp - row["timestamp"])
                if row["action_type"] in ACTIONS_WITH_ITEM_REFERENCE and isinstance(row["fake_impressions"], str):
                    impressions = row["fake_impressions"].split("|")
                    if row["reference"] in impressions and not (event_num == 0 and action_type == "clickout_item"):
                        obs[f"{action_type}_rank_{event_num:02d}"] = impressions.index(row["reference"])

    for event_num, row in enumerate(events_interaction[::-1][:10]):
        if isinstance(row["fake_impressions"], str) and not (event_num == 0 and row["action_type"] == "clickout item"):
            impressions = row["fake_impressions"].split("|")
            if row["reference"] in impressions:
                obs[f"all_events_interaction_rank_{event_num:02d}"] = impressions.index(row["reference"])

    if "index_clicked" in obs:
        all_obs.append(obs)

df_all = pd.DataFrame.from_records(all_obs)

# price_cols = [col for col in df_all.columns if col.startswith("price_")]
# item_id_cols = [col for col in df_all.columns if col.startswith("item_id_impressions_clickout_item")]
cols = [col for col in df_all.columns if col not in ("user_id", "index_clicked")]
X = df_all[cols].fillna(0)
y = df_all["index_clicked"].values
print(list(X.shape))

# +
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor, SymbolicClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

est = make_pipeline(
    make_union(
        SymbolicTransformer(
            parsimony_coefficient=0.001,
            random_state=1,
            function_set=["add", "sub", "mul", "div", "min", "max", "sqrt", "log", "abs", "neg", "inv"],
            verbose=True,
        )
    ),
    LGBMClassifier(n_estimators=20),
)

est = LGBMClassifier(n_estimators=40, importance_type="gain")
# est = MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, verbose=True)
X_tr, X_va, y_tr, y_va = train_test_split(X, y)
est.fit(X_tr, y_tr)

va_pred = est.predict(X_va)
print((va_pred == y_va).mean())

tr_pred = est.predict(X_tr)
print((tr_pred == y_tr).mean())
# -

va_pred_proba = est.predict_proba(X_va)
va_pred_proba

mrr = (1 / (np.argsort(-va_pred_proba, axis=1)[np.arange(va_pred_proba.shape[0]), y_va] + 1)).mean()
mrr

(1 == (np.argsort(-va_pred_proba, axis=1)[np.arange(va_pred_proba.shape[0]), y_va] + 1)).mean()


# +
def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)


logical = make_function(function=_logical, name="logical", arity=4)


def _eq(x1, x2):
    return np.where(x1 == x2, 1, 0)


eq = make_function(function=_eq, name="eq", arity=2)


est = SymbolicRegressor(
    parsimony_coefficient=0.002,
    random_state=1,
    population_size=10000,
    function_set=["add", "sub", "mul", "div", "min", "max", logical, eq],  # , 'sqrt', 'log', 'abs', 'neg', 'inv'],
    feature_names=X.columns,
    n_jobs=1,
    verbose=True,
)
X_tr, X_va, y_tr, y_va = train_test_split(X, y)
est.fit(X_tr, y_tr)

va_pred = est.predict(X_va)
print((va_pred == y_va).mean())

tr_pred = est.predict(X_tr)
print((tr_pred == y_tr).mean())
# -

dot_data = est._program.export_graphviz()
# dot_data = [p for p in est.steps[0][1].transformer_list[0][1]._programs[0] if p][6].export_graphviz()
graph = graphviz.Source(dot_data)
graph

fimp = list(zip(X.columns, est.feature_importances_))
df_imp = pd.DataFrame.from_records(fimp, columns=["col", "imp"])
df_imp.sort_values("imp", ascending=False)

np.log1p(X.max().max())

for user_id in [
    "FQRBCER0ZLL2" "Q96SCXXTDFIY",
    "16P4IKHQOFH9",
    "9IX5ZDR4ILCT",
    "SDIJIN0AB10V",
    "9E2B0S64W9Q1",
    "WK3LAGIG68D1",
    "C68A4O3K23SS",
    "4L9AHS5G0FDC",
    "6W1AAZIPI05W",
    "S5U9A8JTZ3WF",
    "6YTGSXN6TGBL",
    "29TI7SS2CSQ8",
]:

    clickouts = df[(df["user_id"] == user_id) & (df["action_type"] == "clickout item")]
    print(clickouts.shape[0], clickouts["reference"].nunique())

clickouts_num = (
    df[df["action_type"] == "clickout item"].groupby("user_id")["session_id"].count().sort_values(ascending=False)
)
clickouts_num.value_counts()
