import numpy as np
import datatable as dt
from recsys.log_utils import get_logger
from tqdm import tqdm

logger = get_logger()

logger.info("Starting splitting")

df = dt.fread("../../data/events_sorted_trans_all.csv")
filenames = []
for i in tqdm(range(df.shape[0])):
    if (df[i, "is_val"] == False) and (df[i, "is_test"] == False):
        find = int(df[i, "clickout_id"]) % 25
        filenames.append(f"01_train_{find:04d}.csv")
    elif (df[i, "is_val"] == True) and (df[i, "is_test"] == False):
        find = int(df[i, "clickout_id"]) % 2
        filenames.append(f"02_val_{find:04d}.csv")
    elif df[i, "is_test"] == True:
        find = int(df[i, "clickout_id"]) % 4
        filenames.append(f"03_test_{find:04d}.csv")
    else:
        raise (ValueError("Shouldn't happen"))

filenames = np.array(filenames)

for filename in tqdm(set(filenames)):
    df[np.where(filenames == filename)[0], :].to_csv("../../data/proc/raw_csv/" + filename)

logger.info("End splitting")
