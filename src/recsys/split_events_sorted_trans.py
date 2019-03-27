from csv import DictReader, DictWriter

from tqdm import tqdm

header = []

outputs = [
    DictWriter(
        open("../../data/events_sorted_trans_chunks/raw_csv/events_sorted_trans_{:04d}.csv".format(chunk_id), "wt"),
        fieldnames=header,
    )
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
