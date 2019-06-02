import pandas as pd
from joblib import Parallel, delayed
from recsys.log_utils import get_logger

logger = get_logger()

logger.info("Starting splitting")


def get_filename(row):
    if (row["is_val"] == "0") and (row["is_test"] == "0"):
        find = int(row["clickout_id"]) % 25
        return f"01_train_{find}.csv"
    elif (row["is_val"] == "1") and (row["is_test"] == "0"):
        find = 1
        return f"02_val_{find}.csv"
    elif row["is_test"] == "1":
        find = int(row["clickout_id"]) % 4
        return f"03_test_{find}.csv"

logger.info("Reading")
df = pd.read_csv("../../data/events_sorted_trans_all.csv")

logger.info("Assigning filename")
df["filename"] = df.apply(get_filename, axis=1)

def save_pd(df, file):
    df.to_csv(file, index=False)
    return 1

logger.info("Saving")
Parallel(n_jobs=30)(delayed(save_pd)(df_chunk, filename) for filename, df_chunk in df.groupby("filename"))

logger.info("End splitting")

