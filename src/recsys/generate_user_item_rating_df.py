from collections import defaultdict, OrderedDict
from csv import DictReader, DictWriter

from tqdm import tqdm


if __name__ == "__main__":
    inp = open("../../data/events_sorted.csv")
    dr = DictReader(inp)
    out = open("../../data/events_sorted_user_item_rating.csv", "wt")
    # keeps track of item CTR
    all_obs = []
    first_row = True
    for clickout_id, row in enumerate(tqdm(dr)):
        user_id = row["user_id"]
        if row["action_type"] == "clickout item":
            item_ids = row["impressions"].split("|")
            prices = list(map(int, row["prices"].split("|")))
            # create training data
            for rank, (item_id, price) in enumerate(zip(item_ids, prices)):
                obs = OrderedDict()
                obs["src"] = row["src"]
                obs["is_test"] = row["is_test"]
                obs["clickout_id"] = clickout_id
                obs["rank"] = rank
                obs["user_id"] = user_id
                obs["item_id"] = item_id
                obs["timestamp"] = row["timestamp"]
                obs["was_clicked"] = int(row["reference"] == item_id)

                if first_row:
                    dw = DictWriter(out, fieldnames=obs.keys())
                    dw.writeheader()
                    first_row = False

                dw.writerow(obs)

    inp.close()
    out.close()
