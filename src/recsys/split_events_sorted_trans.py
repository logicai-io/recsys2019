from csv import DictReader, DictWriter

from recsys.log_utils import get_logger
from tqdm import tqdm

logger = get_logger()

logger.info("Starting splitting")

header = []

# split the data so that train/val/test are continguous
outputs = [
    DictWriter(open("../../data/proc/raw_csv/01_train_{:04d}.csv".format(chunk_id), "wt"), fieldnames=header)
    for chunk_id in range(20)
] + [
    DictWriter(open("../../data/proc/raw_csv/02_val_{:04d}.csv".format(chunk_id), "wt"), fieldnames=header)
    for chunk_id in range(1)
] + [
    DictWriter(open("../../data/proc/raw_csv/03_test_{:04d}.csv".format(chunk_id), "wt"), fieldnames=header)
    for chunk_id in range(9)
]

reader = DictReader(open("../../data/events_sorted_trans.csv"))

for i, row in tqdm(enumerate(reader)):
    if i == 0:
        for output in outputs:
            output.fieldnames = row.keys()
            output.writeheader()
    if (row["is_val"] == "0") and (row["is_test"] == "0"):
        find = int(row["clickout_id"]) % 20
        outputs[find].writerow(row)
    elif (row["is_val"] == "1") and (row["is_test"] == "0"):
        outputs[20].writerow(row)
    elif (row["is_test"] == "1"):
        find = 20 + int(row["clickout_id"]) % 10
        outputs[find].writerow(row)
    else:
        raise ValueError("Shouldn't happen")

logger.info("Stop splitting")
