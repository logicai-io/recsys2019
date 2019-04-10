import json
from itertools import groupby

import pandas as pd
from tqdm import tqdm

print("Reading")
df = pd.read_csv("../../../data/events_sorted.csv")

print("Filtering")
clickouts_df = df[(df["action_type"] == "clickout item") & (~df["reference"].isnull())]
clickouts_df["reference_int"] = clickouts_df["reference"].map(lambda x: int(x))
clickouts_df["impressions_list"] = (
    clickouts_df["impressions"].fillna("").str.split("|").map(lambda xs: [int(x) for x in xs if x != ""])
)
clickouts = clickouts_df[
    ["user_id", "session_id", "impressions", "impressions_list", "reference_int", "timestamp"]
].to_dict(orient="records")
clickouts = sorted(clickouts, key=lambda x: (x["user_id"], x["timestamp"]))

users_all = []
for user_id, sessions in tqdm(groupby(clickouts, lambda x: x["user_id"])):
    sessions_all = []
    for session_id, session in groupby(sessions, lambda x: x["impressions"]):
        session = list(session)
        first_timestamp = session[0]["timestamp"]
        indices = [
            (row["timestamp"] - first_timestamp, row["impressions_list"].index(row["reference_int"]))
            for row in session
            if row["reference_int"] in row["impressions_list"]
        ]
        sessions_all.append(indices)
    users_all.append(sessions_all)

with open("click_indices.ndjson", "wt") as out:
    for user in tqdm(users_all):
        out.write(json.dumps(user) + "\n")

users_all = []
for user_id, sessions in tqdm(groupby(clickouts, lambda x: x["user_id"])):
    sessions_all = []
    for session_id, session in groupby(sessions, lambda x: x["impressions"]):
        session = list(session)
        first_timestamp = session[0]["timestamp"]
        indices = [
            (row["timestamp"], row["impressions_list"].index(row["reference_int"]))
            for row in session
            if row["reference_int"] in row["impressions_list"]
        ]
        sessions_all.append(indices)
    users_all.append(sessions_all)

with open("click_indices.ndjson", "wt") as out:
    for user in tqdm(users_all):
        out.write(json.dumps(user) + "\n")
