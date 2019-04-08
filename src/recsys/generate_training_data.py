import json
from collections import defaultdict
from csv import DictReader, DictWriter

import click
from recsys.jaccard_sim import ItemPriceSim, JaccardItemSim
from recsys.log_utils import get_logger

# TODO: check usage of impressions_hash vs impressions_raw
# impressions_hash is an old way

ACTION_SHORTENER = {
    "change of sort order": "a",
    "clickout item": "b",
    "filter selection": "c",
    "interaction item deals": "d",
    "interaction item rating": "j",
    "interaction item image": "e",
    "interaction item info": "f",
    "search for destination": "g",
    "search for item": "h",
    "search for poi": "i",
}

ACTIONS_WITH_ITEM_REFERENCE = {
    "search for item",
    "interaction item info",
    "interaction item image",
    "interaction item deals",
    "interaction item rating",
    "clickout item",
}

logger = get_logger()


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


def add_to_set(acc, key, value):
    acc[key].add(value)
    return True


def set_key(acc, key, value):
    acc[key] = value
    return True


def set_key_if_new(acc, key, value):
    if key not in acc:
        acc[key] = value
    return True


def set_nested_key(acc, key1, key2, value):
    acc[key1][key2] = value
    return acc


def add_one_nested_key(acc, key1, key2):
    acc[key1][key2] += 1
    return acc


def append_to_list(acc, key, value):
    acc[key].append(value)
    return True


def append_to_list_not_null(acc, key, value):
    if value:
        acc[key].append(value)
    return True


def diff_ts(acc, current_ts):
    new_acc = acc.copy()
    for key in new_acc.keys():
        new_acc[key] = current_ts - acc[key]
    return new_acc


def tryint(s):
    try:
        return int(s)
    except:
        return 0


def diff(seq):
    new_seq = []
    for n in range(len(seq) - 1):
        new_seq.append(seq[n + 1] - seq[n])
    return new_seq


acc_dict = {}

accumulators = [
    StatsAcc(
        name="identical_impressions_item_clicks",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(lambda: defaultdict(int)),
        updater=lambda acc, row: add_one_nested_key(acc, row["impressions_hash"], row["reference"]),
        get_stats_func=lambda acc, row, item: acc[row["impressions_hash"]][item["item_id"]],
    ),
    StatsAcc(
        name="identical_impressions_item_clicks2",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(lambda: defaultdict(int)),
        updater=lambda acc, row: add_one_nested_key(acc, row["impressions_raw"], row["reference"]),
        get_stats_func=lambda acc, row, item: acc[row["impressions_raw"]][item["item_id"]],
    ),
    StatsAcc(
        name="is_impression_the_same",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(str),
        updater=lambda acc, row: set_key(acc, row["user_id"], row["impressions_hash"]),
        get_stats_func=lambda acc, row, item: acc.get(row["user_id"]) == row["impressions_hash"],
    ),
    StatsAcc(
        name="is_impression_the_same2",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(str),
        updater=lambda acc, row: set_key(acc, row["user_id"], row["impressions_raw"]),
        get_stats_func=lambda acc, row, item: acc.get(row["user_id"]) == row["impressions_raw"],
    ),
    StatsAcc(
        name="last_10_actions",
        filter=lambda row: True,
        acc=defaultdict(list),
        updater=lambda acc, row: append_to_list(acc, row["user_id"], ACTION_SHORTENER[row["action_type"]]),
        get_stats_func=lambda acc, row, item: "".join(["q"] + acc[row["user_id"]] + ["x"]),
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
        name="last_poi",
        filter=lambda row: row["action_type"] == "search for poi",
        acc=defaultdict(lambda: "UNK"),
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: acc.get(row["user_id"], "UNK"),
    ),
    StatsAcc(
        name="last_poi_item_clicks",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(
            acc, (acc_dict["last_poi"].acc[row["user_id"]], row["reference"])
        ),
        get_stats_func=lambda acc, row, item: acc[(acc_dict["last_poi"].acc[row["user_id"]], item["item_id"])],
    ),
    StatsAcc(
        name="last_poi_item_impressions",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_keys_by_one(
            acc, [(acc_dict["last_poi"].acc[row["user_id"]], item_id) for item_id in row["impressions"]]
        ),
        get_stats_func=lambda acc, row, item: acc[(acc_dict["last_poi"].acc[row["user_id"]], item["item_id"])],
    ),
    StatsAcc(
        name="last_item_index",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(list),
        updater=lambda acc, row: append_to_list_not_null(acc, row["user_id"], row["index_clicked"]),
        get_stats_func=lambda acc, row, item: acc[row["user_id"]][-1] - item["rank"] if acc[row["user_id"]] else -1000,
    ),
    StatsAcc(
        name="last_item_index_same_view",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(list),
        updater=lambda acc, row: append_to_list_not_null(
            acc, (row["user_id"], row["impressions_raw"]), row["index_clicked"]
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], row["impressions_raw"])][-1] - item["rank"]
        if acc[(row["user_id"], row["impressions_raw"])]
        else -1000,
    ),
    StatsAcc(
        name="last_event_ts",
        filter=lambda row: True,
        acc=defaultdict(lambda: defaultdict(int)),
        updater=lambda acc, row: set_nested_key(
            acc, row["user_id"], ACTION_SHORTENER[row["action_type"]], row["timestamp"]
        ),
        get_stats_func=lambda acc, row, item: json.dumps(diff_ts(acc[row["user_id"]], row["timestamp"])),
    ),
    StatsAcc(
        name="clickout_user_item_clicks",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
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
        updater=lambda acc, row: increment_key_by_one(acc, row["reference"]),
        get_stats_func=lambda acc, row, item: acc[item["item_id"]],
    ),
    StatsAcc(
        name="clickout_item_platform_clicks",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["reference"], row["platform"])),
        get_stats_func=lambda acc, row, item: acc[(item["item_id"], row["platform"])],
    ),
    StatsAcc(
        name="clickout_item_impressions",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_keys_by_one(acc, row["impressions"]),
        get_stats_func=lambda acc, row, item: acc[item["item_id"]],
    ),
    StatsAcc(
        name="clickout_user_item_impressions",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_keys_by_one(
            acc, [(row["user_id"], item_id) for item_id in row["impressions"]]
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
    ),
    StatsAcc(
        name="was_interaction_img",
        filter=lambda row: row["action_type"] == "interaction item image",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
    ),
    StatsAcc(
        name="interaction_img_diff_ts",
        filter=lambda row: row["action_type"] == "interaction item image",
        acc={},
        updater=lambda acc, row: set_key(acc, (row["user_id"], row["reference"]), row["timestamp"]),
        get_stats_func=lambda acc, row, item: acc.get((row["user_id"], item["item_id"]), item["timestamp"])
        - item["timestamp"],
    ),
    StatsAcc(
        name="interaction_img_freq",
        filter=lambda row: row["action_type"] == "interaction item image",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
    ),
    StatsAcc(
        name="was_interaction_deal",
        filter=lambda row: row["action_type"] == "interaction item deals",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
    ),
    StatsAcc(
        name="interaction_deal_freq",
        filter=lambda row: row["action_type"] == "interaction item deals",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
    ),
    StatsAcc(
        name="was_interaction_rating",
        filter=lambda row: row["action_type"] == "interaction item rating",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
    ),
    StatsAcc(
        name="interaction_rating_freq",
        filter=lambda row: row["action_type"] == "interaction item rating",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
    ),
    StatsAcc(
        name="was_interaction_info",
        filter=lambda row: row["action_type"] == "interaction item info",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
    ),
    StatsAcc(
        name="interaction_info_freq",
        filter=lambda row: row["action_type"] == "interaction item info",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
    ),
    StatsAcc(
        name="was_item_searched",
        filter=lambda row: row["action_type"] == "search for item",
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
        get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
    ),
    StatsAcc(
        name="last_filter",
        filter=lambda row: row["action_type"] in ("filter selection", "search for destination", "search for poi"),
        acc={},
        updater=lambda acc, row: set_key(acc, row["user_id"], row["current_filters"]),
        get_stats_func=lambda acc, row, item: acc.get(row["user_id"], ""),
    ),
    StatsAcc(
        name="user_item_interactions_list",
        filter=lambda row: row["action_type"] in ACTIONS_WITH_ITEM_REFERENCE,
        acc=defaultdict(set),
        updater=lambda acc, row: add_to_set(acc, row["user_id"], tryint(row["reference"])),
        get_stats_func=lambda acc, row, item: list(acc.get(row["user_id"], [])),
    ),
    StatsAcc(
        name="user_item_session_interactions_list",
        filter=lambda row: row["action_type"] in ACTIONS_WITH_ITEM_REFERENCE,
        acc=defaultdict(set),
        updater=lambda acc, row: add_to_set(acc, (row["user_id"], row["session_id"]), tryint(row["reference"])),
        get_stats_func=lambda acc, row, item: list(acc.get((row["user_id"], row["session_id"]), [])),
    ),
    StatsAcc(
        name="user_rank_preference",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["index_clicked"])),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["rank"])],
    ),
    StatsAcc(
        name="user_session_rank_preference",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["session_id"], row["index_clicked"])),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], row["session_id"], item["rank"])],
    ),
    StatsAcc(
        name="user_impression_rank_preference",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(
            acc, (row["user_id"], row["impressions_hash"], row["index_clicked"])
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], row["impressions_hash"], item["rank"])],
    ),
    StatsAcc(
        name="user_impression_rank_preference2",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(
            acc, (row["user_id"], row["impressions_raw"], row["index_clicked"])
        ),
        get_stats_func=lambda acc, row, item: acc[(row["user_id"], row["impressions_raw"], item["rank"])],
    ),
    StatsAcc(
        name="clickout_prices_list",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(set),
        updater=lambda acc, row: add_to_set(acc, row["user_id"], row["price_clicked"]),
        get_stats_func=lambda acc, row, item: list(acc[row["user_id"]]),
    ),
    StatsAcc(
        name="interaction_item_image_item_last_timestamp",
        filter=lambda row: row["action_type"] == "interaction item image",
        acc={},
        updater=lambda acc, row: set_key(
            acc, (row["user_id"], row["reference"], "interaction item image"), row["timestamp"]
        ),
        get_stats_func=lambda acc, row, item: min(
            row["timestamp"] - acc.get((row["user_id"], item["item_id"], "interaction item image"), 0), 1000000
        ),
    ),
    StatsAcc(
        name="clickout_item_item_last_timestamp",
        filter=lambda row: row["action_type"] == "clickout item",
        acc={},
        updater=lambda acc, row: set_key(acc, (row["user_id"], row["reference"], "clickout item"), row["timestamp"]),
        get_stats_func=lambda acc, row, item: min(
            row["timestamp"] - acc.get((row["user_id"], item["item_id"], "clickout item"), 0), 1000000
        ),
    ),
    StatsAcc(
        name="clickout_time_since_first_impression",
        filter=lambda row: row["action_type"] == "clickout item",
        acc={},
        updater=lambda acc, row: set_key_if_new(acc, (row["session_id"], row["impressions_raw"]), row["timestamp"]),
        get_stats_func=lambda acc, row, item: row["timestamp"]
        - acc.get((row["session_id"], row["impressions_raw"]), row["timestamp"] + 1),
    ),
    StatsAcc(
        name="clickout_time_item_since_first_impression",
        filter=lambda row: row["action_type"] == "clickout item",
        acc={},
        updater=lambda acc, row: set_key_if_new(
            acc, (row["session_id"], row["impressions_raw"], row["reference"]), row["timestamp"]
        ),
        get_stats_func=lambda acc, row, item: row["timestamp"]
        - acc.get((row["session_id"], item["item_id"], row["impressions_raw"]), row["timestamp"] + 1),
    ),
    StatsAcc(
        name="interaction_user_item_image_item_since_first_timestamp",
        filter=lambda row: row["action_type"] == "interaction item image",
        acc={},
        updater=lambda acc, row: set_key_if_new(acc, (row["user_id"], row["reference"]), row["timestamp"]),
        get_stats_func=lambda acc, row, item: row["timestamp"]
        - acc.get((row["user_id"], item["item_id"]), row["timestamp"] + 1),
    ),
    StatsAcc(
        name="interaction_user_item_info_since_first_timestamp",
        filter=lambda row: row["action_type"] == "interaction item info",
        acc={},
        updater=lambda acc, row: set_key_if_new(acc, (row["user_id"], row["reference"]), row["timestamp"]),
        get_stats_func=lambda acc, row, item: row["timestamp"]
        - acc.get((row["user_id"], item["item_id"]), row["timestamp"] + 1),
    ),
] + [
    StatsAcc(
        name="{}_count".format(action_type.replace(" ", "_")),
        filter=lambda row: row["action_type"] == action_type,
        acc=defaultdict(int),
        updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], action_type)),
        get_stats_func=lambda acc, row, item: acc.get((row["user_id"], action_type), 0),
    )
    for action_type in ["filter selection"]
]

for acc in accumulators:
    acc_dict[acc.name] = acc


class FeatureGenerator:
    def __init__(self, limit, parallel=True):
        self.limit = limit
        self.jacc_sim = JaccardItemSim(path="../../data/item_metadata_map.joblib")
        self.price_sim = ItemPriceSim(path="../../data/item_prices.joblib")

    def calculate_features_per_item(self, clickout_id, item_id, max_price, mean_price, price, rank, row):
        obs = row.copy()
        obs["item_id"] = item_id
        obs["item_id_clicked"] = row["reference"]
        obs["was_clicked"] = int(row["reference"] == item_id)
        obs["clickout_id"] = clickout_id
        obs["rank"] = rank
        obs["price"] = price
        obs["price_vs_max_price"] = max_price - price
        obs["price_vs_mean_price"] = price / mean_price
        for acc in accumulators:
            obs[acc.name] = acc.get_stats(row, obs)

        self.calculate_similarity_features(item_id, obs)
        self.calculate_indices_features(obs, rank)
        self.calculate_price_similarity(obs, price)

        del obs["impressions"]
        del obs["impressions_hash"]
        del obs["impressions_raw"]
        del obs["prices"]
        del obs["action_type"]
        return obs

    def calculate_price_similarity(self, obs, price):
        if not obs["clickout_prices_list"]:
            output = 1000
        else:
            diff = [abs(p - price) for p in obs["clickout_prices_list"]]
            output = sum(diff) / len(diff)
        obs["avg_price_similarity"] = output
        del obs["clickout_prices_list"]

    def calculate_similarity_features(self, item_id, obs):
        obs["item_similarity_to_last_clicked_item"] = self.jacc_sim.two_items(obs["last_item_clickout"], int(item_id))
        obs["avg_similarity_to_interacted_items"] = self.jacc_sim.list_to_item(
            obs["user_item_interactions_list"], int(item_id)
        )
        obs["avg_similarity_to_interacted_session_items"] = self.jacc_sim.list_to_item(
            obs["user_item_session_interactions_list"], int(item_id)
        )
        obs["avg_price_similarity_to_interacted_items"] = self.price_sim.list_to_item(
            obs["user_item_interactions_list"], int(item_id)
        )
        obs["avg_price_similarity_to_interacted_session_items"] = self.price_sim.list_to_item(
            obs["user_item_session_interactions_list"], int(item_id)
        )

    def calculate_indices_features(self, obs, rank):
        last_n = 5
        last_indices_raw = acc_dict["last_item_index_same_view"].acc[(obs["user_id"], obs["impressions_raw"])]
        last_indices = [-100] * last_n + last_indices_raw
        last_indices = last_indices[-last_n:]
        diff_last_indices = diff(last_indices + [rank])
        for n in range(1, last_n + 1):
            obs["last_index_{}".format(n)] = last_indices[-n]
            obs["last_index_diff_{}".format(n)] = diff_last_indices[-n]
        n_consecutive = FeatureGenerator.calculate_n_consecutive_clicks(last_indices_raw, rank)
        obs["n_consecutive_clicks"] = n_consecutive

    @staticmethod
    def calculate_n_consecutive_clicks(last_indices_raw, rank):
        n_consecutive = 0
        for n in range(1, len(last_indices_raw) + 1):
            if last_indices_raw[-n] == rank:
                n_consecutive += 1
            else:
                break
        return n_consecutive

    def generate_features(self):
        logger.info("Starting feature generation")
        inp = open("../../data/events_sorted.csv")
        dr = DictReader(inp)
        out = open("../../data/events_sorted_trans.csv", "wt")
        first_row = True
        for clickout_id, row in enumerate(dr):
            if clickout_id % 100000 == 0:
                logger.info(clickout_id)
            if self.limit and clickout_id > self.limit:
                break
            row["timestamp"] = int(row["timestamp"])
            if row["action_type"] == "clickout item":
                row["impressions_raw"] = row["impressions"]
                row["impressions"] = row["impressions"].split("|")
                row["impressions_hash"] = "|".join(sorted(row["impressions"]))
                row["index_clicked"] = (
                    row["impressions"].index(row["reference"]) if row["reference"] in row["impressions"] else None
                )
                prices = list(map(int, row["prices"].split("|")))
                row["price_clicked"] = prices[row["index_clicked"]] if row["index_clicked"] else 0
                max_price = max(prices)
                mean_price = sum(prices) / len(prices)

                for rank, (item_id, price) in enumerate(zip(row["impressions"], prices)):
                    obs = self.calculate_features_per_item(
                        clickout_id, item_id, max_price, mean_price, price, rank, row
                    )
                    if first_row:
                        dw = DictWriter(out, fieldnames=obs.keys())
                        dw.writeheader()
                        first_row = False
                    dw.writerow(obs)

            for acc in accumulators:
                acc.update_acc(row)

        inp.close()
        out.close()
        logger.info("Finished feature generation")


@click.command()
@click.option("--limit", type=int, help="Number of rows to process")
def main(limit):
    feature_generator = FeatureGenerator(limit=limit)
    feature_generator.generate_features()


if __name__ == "__main__":
    main()
