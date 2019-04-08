from csv import DictReader, DictWriter

from recsys.log_utils import get_logger
from tqdm import tqdm

logger = get_logger()

logger.info("Starting splitting")

header = []

# split the data so that train/val/test are continguous
outputs_tr = [
    DictWriter(open("../../data/proc/raw_csv/01_train_{:04d}.csv".format(chunk_id), "wt"), fieldnames=header)
    for chunk_id in range(20)
]
outputs_va= [DictWriter(open("../../data/proc/raw_csv/02_val_0001.csv", "wt"), fieldnames=header)]
outputs_te = [
    DictWriter(open("../../data/proc/raw_csv/03_test_{:04d}.csv".format(chunk_id), "wt"), fieldnames=header)
    for chunk_id in range(10)
]

reader = DictReader(open("../../data/events_sorted_trans.csv"))

for i, row in tqdm(enumerate(reader)):
    if i == 0:
        for output in outputs_tr + outputs_va + outputs_te:
            output.fieldnames = row.keys()
            output.writeheader()
    if (row["is_val"] == "0") and (row["is_test"] == "0"):
        find = int(row["clickout_id"]) % 20
        outputs_tr[find].writerow(row)
    elif (row["is_val"] == "1") and (row["is_test"] == "0"):
        outputs_va[0].writerow(row)
    elif (row["is_test"] == "1"):
        find = int(row["clickout_id"]) % 10
        outputs_te[find].writerow(row)
    else:
        raise ValueError("Shouldn't happen")

logger.info("Stop splitting")
