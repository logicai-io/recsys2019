from collections import defaultdict
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm


def calculate_mean_rec_err(queries, clicks):
    recs = []
    for click, query in zip(clicks, queries):
        if query and click in query:
            rec = 1 / (query.index(click) + 1)
            recs.append(rec)
    return np.array(recs).mean()


df = pd.read_csv("../../data/train_sample_1.csv")
df.sort_values(["timestamp", "user_id"], inplace=True)
df["id"] = np.arange(1, df.shape[0] + 1)

item_metadata = pd.read_csv("../../data/item_metadata.csv", dtype={"item_id": str})


# get max timestamp for clickouts
# this extracts user_id and the timestamp of the last checkout
df["num_action_per_user"] = df.groupby(("user_id", "action_type")).cumcount(ascending=False).values
df["validation"] = (df["num_action_per_user"] == 0) & (df["action_type"] == "clickout item")

# keeps track of item CTR
print("Building CTR stats")
impressions_per_item = defaultdict(int)
clicks_per_item = defaultdict(int)

impressions_per_item_user = defaultdict(int)
clicks_per_item_user = defaultdict(int)

impressions_cum = {}
clicks_cum = {}

impressions_user_cum = {}
clicks_user_cum = {}

for index, row in tqdm(df.iterrows()):
    if row["action_type"] == "clickout item":
        clicked_id = row["reference"]
        user_id = row["user_id"]
        impressions = row["impressions"].split("|")

        # track global impressions and clicks
        impressions_stats = [impressions_per_item[i] for i in impressions]
        impressions_cum[index] = ("|".join(map(str, impressions_stats)))

        clicks_stats = [clicks_per_item[i] for i in impressions]
        clicks_cum[index] = ("|".join(map(str, clicks_stats)))

        for item_id in impressions:
            impressions_per_item[item_id] += 1
        clicks_per_item[clicked_id] += 1

        # track user clicks set
        impressions_stats_per_user = [impressions_per_item_user[(i, user_id)] for i in impressions]
        impressions_user_cum[index] = ("|".join(map(str, impressions_stats_per_user)))

        clicks_stats_per_user = [clicks_per_item_user[(i, user_id)] for i in impressions]
        clicks_user_cum[index] = ("|".join(map(str, clicks_stats_per_user)))

        for item_id in impressions:
            impressions_per_item_user[(item_id, user_id)] += 1
        clicks_per_item_user[(clicked_id, user_id)] += 1


df["impressions_cum_for_item"] = df.index.map(impressions_cum)
df["clicks_cum_for_item"] = df.index.map(clicks_cum)
df["impressions_cum_for_item_user"] = df.index.map(impressions_user_cum)
df["clicks_cum_for_item_user"] = df.index.map(clicks_user_cum)

clickouts = df[df["action_type"] == "clickout item"]

# benchmark
print("Benchmark")
print(calculate_mean_rec_err(clickouts["impressions"].map(lambda x: x.split("|")).to_list(), clickouts["reference"]))

dfs = []

def convert_clickout_to_df(row):
    df_ = pd.DataFrame()
    df_["item_id"] = row["impressions"].split("|")
    df_["id"] = row["id"]
    df_["session_id"] = row["session_id"]
    df_["user_id"] = row["user_id"]
    df_["prices"] = list(map(int, row["prices"].split("|")))
    df_["was_clicked"] = df_["item_id"] == row["reference"]
    df_["rank"] = np.arange(df_.shape[0])
    df_["validation"] = row["validation"]
    df_["platform"] = row["platform"]
    df_["city"] = row["city"]
    df_["device"] = row["device"]
    df_["timestamp"] = row["timestamp"]
    df_["current_filters"] = row["current_filters"]
    df_["impressions_before"] = list(map(int, row["impressions_cum_for_item"].split("|")))
    df_["clicks_before"] = list(map(int, row["clicks_cum_for_item"].split("|")))
    df_["impressions_user_before"] = list(map(int, row["impressions_cum_for_item_user"].split("|")))
    df_["clicks_user_before"] = list(map(int, row["clicks_cum_for_item_user"].split("|")))
    return df_


with Pool(8) as pool:
    dfs = list(pool.imap(convert_clickout_to_df, (row for _, row in tqdm(clickouts.iterrows()))))

df_per_item = pd.concat(dfs, axis=0)
df_per_item = pd.merge(df_per_item, item_metadata, on="item_id", how="left")
df_per_item["properties"].fillna("UNK", inplace=True)

split_timestamp = np.percentile(df_per_item.timestamp, 70)

df_per_item["ctr"] = df_per_item["clicks_before"] / (df_per_item["impressions_before"]+1)

df_train = df_per_item[df_per_item["timestamp"] < split_timestamp]
df_val = df_per_item[(df_per_item["timestamp"] > split_timestamp) & (df_per_item["validation"])]

numerical_features = ["rank", "prices", "impressions_before", "clicks_before", "ctr", "impressions_user_before", "clicks_user_before"]
categorical_features = ["device", "platform"]

vectorizer = ColumnTransformer([
    ('numerical', make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), numerical_features),
    ('categorical', make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder()), categorical_features),
    ('properties', CountVectorizer(preprocessor=lambda x: "UNK" if x!=x else x, tokenizer=lambda x: x.split("|"), min_df=5), "properties")
])

model = make_pipeline(vectorizer, VotingClassifier([
    ('lgb', LGBMClassifier(n_estimators=100, n_jobs=-2)),
    ('mlp1', MLPClassifier(hidden_layer_sizes=(25,), n_iter_no_change=3, verbose=True)),
    ('mlp2', MLPClassifier(hidden_layer_sizes=(50,), n_iter_no_change=3, verbose=True)),
    ('mlp3', MLPClassifier(hidden_layer_sizes=(10,10), n_iter_no_change=3, verbose=True))
], voting="soft"))
model.fit(df_train, df_train["was_clicked"])

train_pred = model.predict_proba(df_train)[:, 1]
print(roc_auc_score(df_train["was_clicked"].values, train_pred))

val_pred = model.predict_proba(df_val)[:, 1]
print(roc_auc_score(df_val["was_clicked"].values, val_pred))

df_val["click_proba"] = val_pred

sessions_items = {}
for session_id, session_df in tqdm(df_val.groupby("id")):
    session_df = session_df.sort_values("click_proba", ascending=False)
    sessions_items[session_id] = session_df.item_id.to_list()

clickouts["predicted_items"] = clickouts["id"].map(sessions_items)

# benchmark
print("Benchmark")
print(calculate_mean_rec_err(
    clickouts[~clickouts["predicted_items"].isnull()]["impressions"].map(lambda x: x.split("|")).to_list(),
    clickouts[~clickouts["predicted_items"].isnull()]["reference"]))

print("Model")
print(calculate_mean_rec_err(clickouts[~clickouts["predicted_items"].isnull()]["predicted_items"].to_list(),
                             clickouts[~clickouts["predicted_items"].isnull()]["reference"]))
