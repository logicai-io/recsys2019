import gzip
import json

import click
import numpy as np
import pandas as pd
from multiprocessing.pool import Pool
from recsys.utils import hash_str_to_int
from tqdm import tqdm


def get_index_clicked(row):
    row["reference"] = str(row["reference"])
    if not row["reference"].isnumeric():
        return "UNK"
    item_id = int(row["reference"])
    try:
        impressions = list(map(int, row["fake_impressions"].split("|")))
    except AttributeError:
        return "UNK"
    if item_id in impressions:
        return str(impressions.index(item_id))
    else:
        return "UNK"


def get_price_clicked(row):
    row["reference"] = str(row["reference"])
    if not row["reference"].isnumeric():
        return 0
    item_id = int(row["reference"])
    try:
        impressions = list(map(int, row["fake_impressions"].split("|")))
        prices = list(map(int, row["fake_prices"].split("|")))
    except AttributeError:
        return 0
    if item_id in impressions:
        return prices[impressions.index(item_id)]
    else:
        return 0


def convert_session_df(df_session):
    session = df_session.to_dict(orient="records")
    session = sorted(session, key=lambda x: x["step_min"])
    if session[-1]["action_type"] != "clickout item":
        return None
    n = len(session) - 1
    sequences = {
        "action type cat": ["NA"] * n,
        "clickout item cat": ["NA"] * n,
        "interaction item info cat": ["NA"] * n,
        "interaction item rating cat": ["NA"] * n,
        "interaction item deals cat": ["NA"] * n,
        "interaction item image cat": ["NA"] * n,
        "search for item num": ["NA"] * n,
        "clickout item num": [-1] * n,
        "interaction item info num": [-1] * n,
        "interaction item rating num": [-1] * n,
        "interaction item deals num": [-1] * n,
        "interaction item image num": [-1] * n,
        "filter selection cat": ["NA"] * n,
        "change of sort order cat": ["NA"] * n,
        "search for destination cat": ["NA"] * n,
        "search for item cat": ["NA"] * n,
        "search for poi cat": ["NA"] * n,
        "count action num": [1] * n,
        "time spent num": [0] * n,
        "timestamp diff num": [0] * n,
    }
    for i, obs in enumerate(session[:-1]):
        sequences["action type cat"][i] = obs["action_type"]
        sequences["time spent num"][i] = obs["timestamp_max"] - obs["timestamp_min"]
        sequences["timestamp diff num"][i] = max(0, obs["timestamp_next_min"] - obs["timestamp_max"])

        sequences["count action num"][i] = obs["timestamp_count"]
        if obs["index_clicked"] != "UNK":
            sequences[obs["action_type"] + " cat"][i] = obs["index_clicked"]
            sequences[obs["action_type"] + " num"][i] = int(obs["index_clicked"])
        else:
            if not obs["reference"].isnumeric():
                sequences[obs["action_type"] + " cat"][i] = obs["reference"]

    session_info = {
        "user_id": session[0]["user_id"],
        "session_id": session[0]["session_id"],
        "step": session[-1]["step_min"],
        "src": session[0]["src"],
        "is_val": session[0]["is_val"],
        "is_test": session[0]["is_test"],
        "platform": session[0]["platform"],
        "city": session[0]["city"],
        "device": session[0]["device"],
        "index_clicked": session[-1]["index_clicked"],
        "final_prices": list(map(int, session[-1]["fake_prices"].split("|"))),
        "final_impressions": list(map(int, session[-1]["fake_impressions"].split("|"))),
        "sequences": sequences,
    }
    return session_info


@click.command()
@click.option("--nrows", type=int, default=None, help="Number of rows from events to use")
def main(nrows):
    df = pd.read_csv("../../../data/events_sorted.csv", nrows=nrows)

    df = df.sort_values(["user_id", "session_id", "step"])
    df["clickout_step_rev"][df["action_type"] != "clickout item"] = 0

    df["timestamp_next"] = df.groupby(["user_id", "session_id"])["timestamp"].shift(-1).fillna(0).astype(np.int)
    df["action_type_prv"] = df.groupby(["user_id", "session_id"])["action_type"].shift(1)
    df["session_id_prv"] = df.groupby(["user_id", "session_id"])["session_id"].shift(1)

    # this is to remove the duplicate actions
    df["break_point"] = (
            (df["reference"] != df["reference"])
            | (df["action_type"] != df["action_type_prv"])
            | (df["session_id"] != df["session_id_prv"])
    )
    df["break_point_cumsum"] = df.groupby(["user_id", "session_id"])["break_point"].cumsum()

    df["index_clicked"] = df.apply(get_index_clicked, axis=1)
    df["price_clicked"] = df.apply(get_price_clicked, axis=1)

    df_no_dup = (
        df.groupby(
            [
                "src",
                "is_val",
                "is_test",
                "user_id",
                "session_id",
                "platform",
                "device",
                "city",
                "action_type",
                "reference",
                "break_point_cumsum",
                "fake_prices",
                "fake_impressions",
                "index_clicked",
                "clickout_step_rev",
            ]
        )
            .agg({"timestamp": ["min", "max", "count"], "timestamp_next": ["max", "min"], "step": ["min"]})
            .reset_index()
    )
    df_no_dup.columns = ["_".join(col).strip("_") for col in df_no_dup.columns.values]
    df_no_dup["hash"] = df_no_dup["user_id"].map(hash_str_to_int) % 2

    pool = Pool(8)
    for src in ["train", "test"]:
        for hash_val in [0, 1]:
            with gzip.open(f"../../../data/lstm/seq_user_session_{src}_hash_{hash_val}.ndjson.gzip", "wt") as out:
                generator = df_no_dup[(df_no_dup["src"] == src) & (df_no_dup["hash"] == hash_val)].groupby(
                    ["user_id", "session_id"]
                )
                save_parallel(generator, out, pool)

            with gzip.open(f"../../../data/lstm/seq_user_{src}_hash_{hash_val}.ndjson.gzip", "wt") as out:
                generator = df_no_dup[(df_no_dup["src"] == src) & (df_no_dup["hash"] == hash_val)].groupby(["user_id"])
                save_parallel(generator, out, pool)
    pool.close()


def save_parallel(generator, out, pool):
    parallel_gen = pool.imap_unordered(
        convert_session_df, (df_session for _, df_session in tqdm(generator)), chunksize=400
    )
    for session_info in parallel_gen:
        if session_info:
            out.writelines(json.dumps(session_info) + "\n")


if __name__ == "__main__":
    main()
