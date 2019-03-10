import json
from collections import defaultdict
from csv import DictReader, DictWriter

import click
from tqdm import tqdm

ACTION_SHORTENER = {
    "change of sort order": "a",
    "clickout item": "b",
    "filter selection": "c",
    "interaction item deals": "d",
    "interaction item image": "e",
    "interaction item info": "f",
    "search for destination": "g",
    "search for item": "h",
    "search for poi": "i",
}


class StatsAcc:
    """
    Example definition

    StatsAcc(filter=lambda row: row.action_type == "clickout_item",
             init_acc=defaultdict(int),
             updater=lambda acc, row: acc[(row.user_id, row.item_id)]+=1)
    """

    def __init__(self, name, filter, acc, updater, get_stats_func):
        self.name = name
        self.filter = filter
        self.acc = acc
        self.updater = updater
        self.get_stats_func = get_stats_func

    def filter(self, row):
        return self.filter(row)

    def update_acc(self, row):
        if self.filter(row):
            self.updater(self.acc, row)

    def get_stats(self, row, item):
        return self.get_stats_func(self.acc, row, item)


def increment_key_by_one(acc, key):
    acc[key] += 1
    return acc


def increment_keys_by_one(acc, keys):
    for key in keys:
        acc[key] += 1
    return acc


def set_key(acc, key, value):
    acc[key] = value
    return True


def set_nested_key(acc, key1, key2, value):
    acc[key1][key2] = value
    return acc


def append_to_list(acc, key, value):
    acc[key].append(value)
    return True


def diff_ts(acc, current_ts):
    new_acc = acc.copy()
    for key in new_acc.keys():
        new_acc[key] = current_ts - acc[key]
    return new_acc


accumulators = [
    StatsAcc(
        name="last_10_actions",
        filter=lambda row: True,
        acc=defaultdict(list),
        updater=lambda acc, row: append_to_list(
            acc, row["user_id"], ACTION_SHORTENER[row["action_type"]]
        ),
        get_stats_func=lambda acc, row, item: "".join(
            ["q"] + acc[row["user_id"]] + ["x"]
        ),
    ),
    StatsAcc(
        name="last_sort_order",
        filter=lambda row: row["action_type"] == "change of sort order",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: acc.get(row["user_id"], "UNK"),
    ),
    StatsAcc(
        name="last_filter_selection",
        filter=lambda row: row["action_type"] == "filter selection",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: acc.get(row["user_id"], "UNK"),
    ),
    StatsAcc(
        name="last_item_index",
        filter=lambda row: row["action_type"] == "clickout item",
        acc={},
        updater=lambda acc, row: set_key(
            acc,
            row["user_id"],
            row["impressions"].index(row["reference"])
            if row["reference"] in row["impressions"]
            else -2,
        ),
        get_stats_func=lambda acc, row, item: acc.get(row["user_id"], -1),
    ),
    StatsAcc(
        name="last_event_ts",
        filter=lambda row: True,
        acc=defaultdict(lambda: defaultdict(int)),
        updater=lambda acc, row: set_nested_key(
            acc, row["user_id"], ACTION_SHORTENER[row["action_type"]], row["timestamp"]
        ),
        get_stats_func=lambda acc, row, item: json.dumps(
            diff_ts(acc[row["user_id"]], row["timestamp"])
        ),
    ),
    StatsAcc(
        name="clickout_user_item_clicks",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(
            acc, (row["user_id"], row["reference"])
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item)],
    ),
    StatsAcc(
        name="last_item_clickout",
        filter=lambda row: row["action_type"] == "clickout item",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: acc.get(row["user_id"], 0),
    ),
    StatsAcc(
        name="clickout_item_clicks",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["reference"])),
        get_stats_func=lambda acc, row, item: acc[item],
    ),
    StatsAcc(
        name="clickout_item_impressions",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_keys_by_one(acc, row["impressions"]),
        get_stats_func=lambda acc, row, item: acc[item],
    ),
    StatsAcc(
        name="clickout_user_item_impressions",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_keys_by_one(
            acc, [(row["user_id"], item_id) for item_id in row["impressions"]]
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item)],
    ),
    StatsAcc(
        name="was_interaction_img",
        filter=lambda row: row["action_type"] == "interaction item image",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item),
    ),
    StatsAcc(
        name="interaction_img_freq",
        filter=lambda row: row["action_type"] == "interaction item image",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(
            acc, (row["user_id"], row["reference"])
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item)],
    ),
    StatsAcc(
        name="was_interaction_deal",
        filter=lambda row: row["action_type"] == "interaction item deals",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item),
    ),
    StatsAcc(
        name="interaction_deal_freq",
        filter=lambda row: row["action_type"] == "interaction item deals",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(
            acc, (row["user_id"], row["reference"])
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item)],
    ),
    StatsAcc(
        name="was_interaction_info",
        filter=lambda row: row["action_type"] == "interaction item info",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item),
    ),
    StatsAcc(
        name="interaction_info_freq",
        filter=lambda row: row["action_type"] == "interaction item info",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(
            acc, (row["user_id"], row["reference"])
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item)],
    ),
    StatsAcc(
        name="was_item_searched",
        filter=lambda row: row["action_type"] == "search for item",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item),
    ),
]


@click.command()
@click.option("--limit", type=int, help="Number of rows to process")
def main(limit):
    inp = open("../../data/events_sorted.csv")
    dr = DictReader(inp)
    out = open("../../data/events_sorted_trans.csv", "wt")
    # keeps track of item CTR
    all_obs = []
    first_row = True
    for clickout_id, row in enumerate(tqdm(dr)):
        if limit and clickout_id > limit:
            break
        user_id = row["user_id"]
        row["timestamp"] = int(row["timestamp"])
        if row["action_type"] == "clickout item":
            row["impressions"] = row["impressions"].split("|")
            prices = list(map(int, row["prices"].split("|")))
            # create training data
            for rank, (item_id, price) in enumerate(zip(row["impressions"], prices)):
                obs = row.copy()
                del obs["impressions"]
                del obs["prices"]
                del obs["action_type"]
                obs["item_id"] = item_id
                obs["item_id_clicked"] = row["reference"]
                obs["was_clicked"] = int(row["reference"] == item_id)
                obs["clickout_id"] = clickout_id
                obs["rank"] = rank
                obs["price"] = price
                for acc in accumulators:
                    obs[acc.name] = acc.get_stats(row, item_id)

                if first_row:
                    dw = DictWriter(out, fieldnames=obs.keys())
                    dw.writeheader()
                    first_row = False

                dw.writerow(obs)

        for acc in accumulators:
            acc.update_acc(row)

    inp.close()
    out.close()


if __name__ == "__main__":
    main()
