import gzip
import hashlib
import multiprocessing
import queue
from csv import DictReader, DictWriter

from recsys.log_utils import get_logger
from tqdm import tqdm


def write_worker(q: queue.Queue, filename):
    with open(filename, "wt") as out:
        header = []
        dw = DictWriter(out, fieldnames=header)
        header_written = False
        while True:
            row = q.get()
            if not header_written:
                dw.fieldnames = row.keys()
                dw.writeheader()
                header_written = True
            dw.writerow(row)


def create_queue_process(filename):
    q = multiprocessing.Queue(maxsize=10000)
    process = multiprocessing.Process(target=write_worker, args=(q, filename))
    process.daemon = True
    process.start()
    return q, process


def hash_str(text):
    return int(hashlib.sha1(text).hexdigest(), 16) % (10 ** 8)


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Starting splitting")

    # split the data so that train/val/test are continguous
    outputs_tr = [
        create_queue_process("../../data/proc/raw_csv/01_train_{:04d}.csv".format(chunk_id)) for chunk_id in range(25)
    ]
    outputs_va = [create_queue_process("../../data/proc/raw_csv/02_val_0001.csv")]
    outputs_te = [
        create_queue_process("../../data/proc/raw_csv/03_test_{:04d}.csv".format(chunk_id)) for chunk_id in range(4)
    ]

    reader = DictReader(open("../../data/events_sorted_trans_all.csv"))

    for i, row in tqdm(enumerate(reader)):
        hashn = hash_str(row["user_id"])
        if (row["is_val"] == "0") and (row["is_test"] == "0"):
            find = hashn % 25
            outputs_tr[find][0].put(row)
        elif (row["is_val"] == "1") and (row["is_test"] == "0"):
            outputs_va[0][0].put(row)
        elif row["is_test"] == "1":
            find = hashn % 4
            outputs_te[find][0].put(row)
        else:
            raise ValueError("Shouldn't happen")

    logger.info("Stop splitting")

    for queue, process in outputs_tr + outputs_va + outputs_te:
        process.terminate()
