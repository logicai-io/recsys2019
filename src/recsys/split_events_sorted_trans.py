from csv import DictReader, DictWriter

from recsys.log_utils import get_logger
from tqdm import tqdm

logger = get_logger()

logger.info("Starting splitting")

header = []

outputs = [
    DictWriter(open("../../data/proc/raw_csv/{:04d}.csv".format(chunk_id), "wt"), fieldnames=header)
    for chunk_id in range(30)
]

reader = DictReader(open("../../data/events_sorted_trans.csv"))

for i, row in tqdm(enumerate(reader)):
    if i == 0:
        for output in outputs:
            output.fieldnames = row.keys()
            output.writeheader()
    find = int(row["clickout_id"]) % 30
    outputs[find].writerow(row)

logger.info("Stop splitting")
