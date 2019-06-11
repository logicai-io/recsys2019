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
import datatable as dt
from tqdm import tqdm
import graphviz

ACTIONS_WITH_ITEM_REFERENCE = {
    "search for item",
    "interaction item info",
    "interaction item image",
    "interaction item deals",
    "interaction item rating",
    "clickout item",
}
# -

df = pd.read_csv("../../../data/events_sorted.csv", nrows=100000) 

all_obs = []
for user_id, df_user in tqdm(df.iloc[:100000].groupby("user_id")):
    obs = {}
    obs["user_id"] = user_id
    interacted_ranks = []
    for row in df_user.to_dict(orient='records'):
        
        action_type = row["action_type"].replace(" ", "_")
        
        if action_type == "clickout_item" and row["clickout_step_rev"] > 1:
            impressions = row["impressions"].split("|")
                
        if action_type == "clickout_item" and row["clickout_step_rev"] == 1:
            impressions = row["impressions"].split("|")
            if row["reference"] not in impressions:
                continue
            obs["index_clicked"] = impressions.index(row["reference"])
            for rank, item_id in enumerate(row["impressions"].split("|")):
                obs["item_id_impressions_clickout_item_{:02d}".format(rank+1)] = int(item_id)
            for rank, price in enumerate(row["prices"].split("|")):
                obs["price_clickout_item_{:02d}".format(rank+1)] = int(price)
        
        if row["clickout_step_rev"] <= 10:
            obs["timestamp_{}_{:02d}".format(action_type, row["clickout_step_rev"])] = "TS="+str(row["timestamp"])
            if row["action_type"] in ACTIONS_WITH_ITEM_REFERENCE:
                try:
                    impressions = row["impressions"].split("|")
                    if row["reference"] in impressions:
                        interacted_ranks.append(impressions.index(row["reference"]))
                except:
                    pass
                        
                obs["item_id_{}_{:02d}".format(action_type, row["clickout_step_rev"])] = "ITEM_ID="+str(row["reference"])
                
    for i, rank in enumerate(interacted_ranks[::-1]):
        obs["interacted_rank_{}".format(i)] = rank

    all_obs.append(obs)

df_all = pd.DataFrame.from_records(all_obs)

price_cols = [col for col in df_all.columns if col.startswith("price_")]
item_id_cols = [col for col in df_all.columns if col.startswith("item_id_impressions_clickout_item")]
df_all = df_all[df_all["index_clicked"] > 0]
X = df_all[price_cols + item_id_cols].fillna(-1)
y = df_all["index_clicked"]

X.dtypes

# +
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

est = make_pipeline(
        SymbolicTransformer(parsimony_coefficient=.001,
                          random_state=1,
                          function_set=['add', 'sub', 'mul', 'div', 'min', 'max'],
                          verbose=True),
        LGBMClassifier()
)

X_tr, X_va, y_tr, y_va = train_test_split(X, y)
est.fit(X_tr, y_tr)

pred = est.predict(X_va)

print((pred == y_va).mean())
# -

dot_data = est._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph

print((pred == y_va.values).mean())

pred


