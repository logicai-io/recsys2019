# +
from collections import defaultdict
from itertools import groupby

import joblib
import pandas as pd
from recsys.utils import group_time
from tqdm import tqdm

print("Reading")
raw_df = pd.read_csv("../../../data/events_sorted.csv")

print("Filtering")

for device in ["desktop", "mobile", "tablet"]:
    clickouts_df = raw_df[
        (raw_df["action_type"] == "clickout item") & (~raw_df["reference"].isnull()) & (raw_df["device"] == device)
    ]
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
                (row["timestamp"], row["impressions_list"].index(row["reference_int"]))
                for row in session
                if row["reference_int"] in row["impressions_list"]
            ]
            sessions_all.append(indices)
        users_all.append(sessions_all)

    diff_dist = defaultdict(list)
    records = []
    for i, user in tqdm(enumerate(users_all)):
        for session in user:
            if len(session) > 1:
                for ((t1, c1), (t2, c2)) in zip(session[:-1], session[1:]):
                    if t2 - t1 <= 120:
                        records.append((c2 - c1, t2 - t1))

    df = pd.DataFrame.from_records(records)
    df.columns = ["click_offset", "diff_time"]
    df["diff_time_grouped"] = df["diff_time"].map(group_time)
    df["one"] = 1

    diff_time_pivot = df.pivot_table(values="one", index="diff_time_grouped", columns="click_offset", aggfunc=len)
    diff_time_pivot.fillna(1.0, inplace=True)
    diff_time_pivot = diff_time_pivot.div(diff_time_pivot.sum(axis=1), axis=0)

    records = []
    for ind in diff_time_pivot.index:
        for col in diff_time_pivot.columns:
            records.append((ind, col, diff_time_pivot.ix[ind, col]))

    diff_time = pd.DataFrame.from_records(records)
    diff_time.columns = ["time", "offset", "prob"]

    d = dict(zip(diff_time.apply(lambda row: (int(row["offset"]), int(row["time"])), axis=1), diff_time["prob"]))

    joblib.dump(d, f"../../../data/click_probs_by_index_and_{device}.joblib")
