import json
from collections import defaultdict

import joblib
from recsys.jaccard_sim import JaccardItemSim, ItemPriceSim
from recsys.log_utils import get_logger
from recsys.utils import group_time

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
ALL_ACTIONS = list(ACTION_SHORTENER.keys())
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

    def __init__(self, name, action_types, acc, updater, get_stats_func):
        self.name = name
        self.action_types = action_types
        self.acc = acc
        self.updater = updater
        self.get_stats_func = get_stats_func

    def filter(self, row):
        return self.action_types(row)

    def update_acc(self, row):
        self.updater(self.acc, row)

    def get_stats(self, row, item):
        return self.get_stats_func(self.acc, row, item)


class ClickSequenceEncoder:
    def __init__(self):
        self.name = "click_index_sequence"
        self.current_impression = {}
        self.sequences = defaultdict(list)
        self.action_types = ["clickout item"]

    def update_acc(self, row):
        if row["action_type"] in self.action_types:
            key = (row["user_id"], row["session_id"])
            if self.current_impression.get(key) == row["impressions_raw"]:
                self.sequences[key][-1].append(row["index_clicked"])
            else:
                self.sequences[key].append([row["index_clicked"]])
            self.current_impression[key] = row["impressions_raw"]

    def get_stats(self, row, item):
        key = (row["user_id"], row["session_id"])
        return json.dumps(self.sequences[key])


class ClickProbabilityClickOffsetTimeOffset:
    def __init__(
        self,
        name="clickout_prob_time_position_offset",
        action_types=None,
        impressions_type="impressions_raw",
        index_col="index_clicked",
    ):
        self.name = name
        self.action_types = action_types
        self.index_col = index_col
        self.impressions_type = impressions_type
        # tracks the impression per user
        self.current_impression = defaultdict(str)
        self.last_timestamp = {}
        self.last_clickout_position = {}
        self.read_probs()

    def read_probs(self):
        self.probs = joblib.load("../../data/click_probs_by_index.joblib")

    def update_acc(self, row):
        self.current_impression[row["user_id"]] = row[self.impressions_type]
        key = (row["user_id"], row[self.impressions_type])
        self.last_timestamp[key] = row["timestamp"]
        self.last_clickout_position[key] = row[self.index_col]

    def get_stats(self, row, item):
        key = (row["user_id"], row[self.impressions_type])

        if row[self.impressions_type] == self.current_impression[row["user_id"]]:
            t1 = self.last_timestamp[key]
            t2 = row["timestamp"]

            c1 = self.last_clickout_position[key]
            c2 = item["rank"]

            timestamp_offset = int(group_time(t2 - t1))
            click_offset = int(c2 - c1)

            key = (click_offset, timestamp_offset)

            if key in self.probs:
                return self.probs[key]
            else:
                try:
                    return self.probs[(click_offset, 120)]
                except KeyError:
                    return self.default_click_prob(item)

        else:
            # TODO fill this with prior distribution for positions
            return self.default_click_prob(item)

    def default_click_prob(self, item):
        probs = {0: 0.3, 1: 0.2, 2: 0.1, 3: 0.07, 4: 0.05, 5: 0.03}
        try:
            return probs[item["rank"]]
        except KeyError:
            return 0.03


class PoiFeatures:
    def __init__(self):
        self.name = "last_poi_features"
        self.action_types = ["search for poi", "clickout item"]
        self.last_poi = defaultdict(lambda: "UNK")
        self.last_poi_clicks = defaultdict(int)
        self.last_poi_impressions = defaultdict(int)

    def update_acc(self, row):
        if row["action_type"] == "search for poi":
            self.last_poi[row["user_id"]] = row["reference"]
        if row["action_type"] == "clickout item":
            self.last_poi_clicks[(self.last_poi[row["user_id"]], row["reference"])] += 1
            for item_id in row["impressions"]:
                self.last_poi_impressions[(self.last_poi[row["user_id"]], item_id)] += 1

    def get_stats(self, row, item):
        output = {}
        output["last_poi"] = self.last_poi[row["user_id"]]
        output["last_poi_item_clicks"] = self.last_poi_clicks[(output["last_poi"], item["item_id"])]
        output["last_poi_item_impressions"] = self.last_poi_impressions[(output["last_poi"], item["item_id"])]
        output["last_poi_ctr"] = output["last_poi_item_clicks"] / (output["last_poi_item_impressions"] + 1)
        return output


class IndicesFeatures:
    def __init__(
        self, action_types=["clickout item"], impressions_type="impressions_raw", index_key="index_clicked", prefix=""
    ):
        self.action_types = action_types
        self.impressions_type = impressions_type
        self.index_key = index_key
        self.last_indices = defaultdict(list)
        self.prefix = prefix

    def update_acc(self, row):
        # TODO: reset list when there is a change of sort order?
        if row["action_type"] in self.action_types and row[self.index_key] >= 0:
            self.last_indices[(row["user_id"], row[self.impressions_type])].append(row[self.index_key])

    def get_stats(self, row, item):
        last_n = 5
        last_indices_raw = self.last_indices[(row["user_id"], row[self.impressions_type])]
        last_indices = [-100] * last_n + last_indices_raw
        last_indices = last_indices[-last_n:]
        diff_last_indices = diff(last_indices + [item["rank"]])
        output = {}
        for n in range(1, last_n + 1):
            output[self.prefix + "last_index_{}".format(n)] = last_indices[-n]
            output[self.prefix + "last_index_diff_{}".format(n)] = diff_last_indices[-n]
        n_consecutive = self._calculate_n_consecutive_clicks(last_indices_raw, item["rank"])
        output[self.prefix + "n_consecutive_clicks"] = n_consecutive
        output[self.prefix + "last_index_diff"] = last_indices[-1] - item["rank"]
        return output

    def _calculate_n_consecutive_clicks(self, last_indices_raw, rank):
        n_consecutive = 0
        for n in range(1, len(last_indices_raw) + 1):
            if last_indices_raw[-n] == rank:
                n_consecutive += 1
            else:
                break
        return n_consecutive


class PriceSimilarity:
    def __init__(self):
        self.action_types = ["clickout item"]
        self.last_prices = defaultdict(list)

    def update_acc(self, row):
        self.last_prices[row["user_id"]].append(row["price_clicked"])

    def get_stats(self, row, item):
        clickout_prices_list = self.last_prices[row["user_id"]]
        if not clickout_prices_list:
            output = 1000
            last_price_diff = 1000
        else:
            diff = [abs(p - item["price"]) for p in list(set(clickout_prices_list))]
            output = sum(diff) / len(diff)
            last_price_diff = clickout_prices_list[-1] - item["price"]
        obs = {}
        obs["avg_price_similarity"] = output
        obs["last_price_diff"] = last_price_diff
        return obs


class PriceFeatures:
    def __init__(self):
        self.action_types = ["clickout item"]

    def update_acc(self, row):
        pass

    def get_stats(self, row, item):
        max_price = max(row["prices"])
        mean_price = sum(row["prices"]) / len(row["prices"])
        obs = {}
        obs["price_vs_max_price"] = max_price - item["price"]
        obs["price_vs_mean_price"] = item["price"] / mean_price
        return obs


class SimilarityFeatures:
    def __init__(self):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.jacc_sim = JaccardItemSim(path="../../data/item_metadata_map.joblib")
        self.poi_sim = JaccardItemSim(path="../../data/item_pois.joblib")
        self.price_sim = ItemPriceSim(path="../../data/item_prices.joblib")
        self.last_item_clickout = defaultdict(int)
        self.user_item_interactions_list = defaultdict(set)
        self.user_item_session_interactions_list = defaultdict(set)

    def update_acc(self, row):
        if row["action_type"] == "clickout item":
            self.last_item_clickout[row["user_id"]] = row["reference"]
        if row["action_type"] in ACTIONS_WITH_ITEM_REFERENCE:
            self.user_item_interactions_list[row["user_id"]].add(tryint(row["reference"]))
            self.user_item_session_interactions_list[(row["user_id"], row["session_id"])].add(tryint(row["reference"]))

    def get_stats(self, row, item):
        user_item_interactions_list = list(self.user_item_interactions_list[row["user_id"]])
        user_item_session_interactions_list = list(
            self.user_item_session_interactions_list[(row["user_id"], row["session_id"])]
        )
        last_item_clickout = self.last_item_clickout[row["user_id"]]
        item_id = int(item["item_id"])
        output = {}
        output["item_similarity_to_last_clicked_item"] = self.jacc_sim.two_items(last_item_clickout, item["item_id"])
        output["avg_similarity_to_interacted_items"] = self.jacc_sim.list_to_item(user_item_interactions_list, item_id)
        output["avg_similarity_to_interacted_session_items"] = self.jacc_sim.list_to_item(
            user_item_session_interactions_list, item_id
        )
        output["avg_price_similarity_to_interacted_items"] = self.price_sim.list_to_item(
            user_item_interactions_list, item_id
        )
        output["avg_price_similarity_to_interacted_session_items"] = self.price_sim.list_to_item(
            user_item_session_interactions_list, item_id
        )
        output["poi_item_similarity_to_last_clicked_item"] = self.poi_sim.two_items(last_item_clickout, item_id)
        output["poi_avg_similarity_to_interacted_items"] = self.poi_sim.list_to_item(
            user_item_interactions_list, item_id
        )
        output["num_pois"] = len(self.poi_sim.imm[item_id])
        return output


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


def get_accumulators():
    accumulators = [
        StatsAcc(
            name="identical_impressions_item_clicks",
            action_types=["clickout item"],
            acc=defaultdict(lambda: defaultdict(int)),
            updater=lambda acc, row: add_one_nested_key(acc, row["impressions_hash"], row["reference"]),
            get_stats_func=lambda acc, row, item: acc[row["impressions_hash"]][item["item_id"]],
        ),
        StatsAcc(
            name="identical_impressions_item_clicks2",
            action_types=["clickout item"],
            acc=defaultdict(lambda: defaultdict(int)),
            updater=lambda acc, row: add_one_nested_key(acc, row["impressions_raw"], row["reference"]),
            get_stats_func=lambda acc, row, item: acc[row["impressions_raw"]][item["item_id"]],
        ),
        StatsAcc(
            name="is_impression_the_same",
            action_types=["clickout item"],
            acc=defaultdict(str),
            updater=lambda acc, row: set_key(acc, row["user_id"], row["impressions_hash"]),
            get_stats_func=lambda acc, row, item: acc.get(row["user_id"]) == row["impressions_hash"],
        ),
        StatsAcc(
            name="last_10_actions",
            action_types=ALL_ACTIONS,
            acc=defaultdict(list),
            updater=lambda acc, row: append_to_list(acc, row["user_id"], ACTION_SHORTENER[row["action_type"]]),
            get_stats_func=lambda acc, row, item: "".join(["q"] + acc[row["user_id"]] + ["x"]),
        ),
        StatsAcc(
            name="last_sort_order",
            action_types=["change of sort order"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: acc.get(row["user_id"], "UNK"),
        ),
        StatsAcc(
            name="last_filter_selection",
            action_types=["filter selection"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: acc.get(row["user_id"], "UNK"),
        ),
        StatsAcc(
            name="last_item_index",
            action_types=["clickout item"],
            acc=defaultdict(list),
            updater=lambda acc, row: append_to_list_not_null(acc, row["user_id"], row["index_clicked"]),
            get_stats_func=lambda acc, row, item: acc[row["user_id"]][-1] - item["rank"]
            if acc[row["user_id"]]
            else -1000,
        ),
        StatsAcc(
            name="last_item_fake_index",
            action_types=ACTIONS_WITH_ITEM_REFERENCE,
            acc=defaultdict(list),
            updater=lambda acc, row: append_to_list_not_null(acc, row["user_id"], row["fake_index_interacted"]),
            get_stats_func=lambda acc, row, item: acc[row["user_id"]][-1] - item["rank"]
            if acc[row["user_id"]]
            else -1000,
        ),
        StatsAcc(
            name="last_clicked_item_position_same_view",
            action_types=["clickout item"],
            acc={},
            updater=lambda acc, row: set_key(acc, (row["user_id"], row["impressions_raw"]), row["index_clicked"]),
            get_stats_func=lambda acc, row, item: item["rank"]
            - acc.get((row["user_id"], row["impressions_raw"]), -1000),
        ),
        StatsAcc(
            name="last_item_index_same_view",
            action_types=["clickout item"],
            acc=defaultdict(list),
            updater=lambda acc, row: append_to_list_not_null(
                acc, (row["user_id"], row["impressions_raw"]), row["index_clicked"]
            ),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], row["impressions_raw"])][-1] - item["rank"]
            if acc[(row["user_id"], row["impressions_raw"])]
            else -1000,
        ),
        StatsAcc(
            name="last_item_index_same_fake_view",
            action_types=ACTIONS_WITH_ITEM_REFERENCE,
            acc=defaultdict(list),
            updater=lambda acc, row: append_to_list_not_null(
                acc, (row["user_id"], row["fake_impressions_raw"]), row["fake_index_interacted"]
            ),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], row["fake_impressions_raw"])][-1] - item["rank"]
            if acc[(row["user_id"], row["fake_impressions_raw"])]
            else -1000,
        ),
        StatsAcc(
            name="last_event_ts",
            action_types=ALL_ACTIONS,
            acc=defaultdict(lambda: defaultdict(int)),
            updater=lambda acc, row: set_nested_key(
                acc, row["user_id"], ACTION_SHORTENER[row["action_type"]], row["timestamp"]
            ),
            get_stats_func=lambda acc, row, item: json.dumps(diff_ts(acc[row["user_id"]], row["timestamp"])),
        ),
        StatsAcc(
            name="clickout_user_item_clicks",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
        ),
        StatsAcc(
            name="last_item_clickout",
            action_types=["clickout item"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: acc.get(row["user_id"], 0),
        ),
        StatsAcc(
            name="clickout_item_clicks",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, row["reference"]),
            get_stats_func=lambda acc, row, item: acc[item["item_id"]],
        ),
        StatsAcc(
            name="clickout_item_platform_clicks",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["reference"], row["platform"])),
            get_stats_func=lambda acc, row, item: acc[(item["item_id"], row["platform"])],
        ),
        StatsAcc(
            name="clickout_item_impressions",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_keys_by_one(acc, row["impressions"]),
            get_stats_func=lambda acc, row, item: acc[item["item_id"]],
        ),
        StatsAcc(
            name="clickout_user_item_impressions",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_keys_by_one(
                acc, [(row["user_id"], item_id) for item_id in row["impressions"]]
            ),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
        ),
        StatsAcc(
            name="was_interaction_img",
            action_types=["interaction item image"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
        ),
        StatsAcc(
            name="interaction_img_diff_ts",
            action_types=["interaction item image"],
            acc={},
            updater=lambda acc, row: set_key(acc, (row["user_id"], row["reference"]), row["timestamp"]),
            get_stats_func=lambda acc, row, item: acc.get((row["user_id"], item["item_id"]), item["timestamp"])
            - item["timestamp"],
        ),
        StatsAcc(
            name="interaction_img_freq",
            action_types=["interaction item image"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
        ),
        StatsAcc(
            name="was_interaction_deal",
            action_types=["interaction item deals"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
        ),
        StatsAcc(
            name="interaction_deal_freq",
            action_types=["interaction item deals"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
        ),
        StatsAcc(
            name="was_interaction_rating",
            action_types=["interaction item rating"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
        ),
        StatsAcc(
            name="interaction_rating_freq",
            action_types=["interaction item rating"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
        ),
        StatsAcc(
            name="was_interaction_info",
            action_types=["interaction item info"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
        ),
        StatsAcc(
            name="interaction_info_freq",
            action_types=["interaction item info"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
        ),
        StatsAcc(
            name="was_item_searched",
            action_types=["search for item"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: int(acc.get(row["user_id"]) == item["item_id"]),
        ),
        StatsAcc(
            name="last_filter",
            action_types=["filter selection", "search for destination", "search for poi"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["current_filters"]),
            get_stats_func=lambda acc, row, item: acc.get(row["user_id"], ""),
        ),
        StatsAcc(
            name="user_item_interactions_list",
            action_types=ACTIONS_WITH_ITEM_REFERENCE,
            acc=defaultdict(set),
            updater=lambda acc, row: add_to_set(acc, row["user_id"], tryint(row["reference"])),
            get_stats_func=lambda acc, row, item: list(acc.get(row["user_id"], [])),
        ),
        StatsAcc(
            name="user_item_session_interactions_list",
            action_types=ACTIONS_WITH_ITEM_REFERENCE,
            acc=defaultdict(set),
            updater=lambda acc, row: add_to_set(acc, (row["user_id"], row["session_id"]), tryint(row["reference"])),
            get_stats_func=lambda acc, row, item: list(acc.get((row["user_id"], row["session_id"]), [])),
        ),
        StatsAcc(
            name="user_rank_preference",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["index_clicked"])),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["rank"])],
        ),
        StatsAcc(
            name="user_fake_rank_preference",
            action_types=ACTIONS_WITH_ITEM_REFERENCE,
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["fake_index_interacted"])),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["rank"])],
        ),
        StatsAcc(
            name="user_session_rank_preference",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(
                acc, (row["user_id"], row["session_id"], row["index_clicked"])
            ),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], row["session_id"], item["rank"])],
        ),
        StatsAcc(
            name="user_impression_rank_preference",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(
                acc, (row["user_id"], row["impressions_hash"], row["index_clicked"])
            ),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], row["impressions_hash"], item["rank"])],
        ),
        StatsAcc(
            name="interaction_item_image_item_last_timestamp",
            action_types=["interaction item image"],
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
            action_types=["clickout item"],
            acc={},
            updater=lambda acc, row: set_key(
                acc, (row["user_id"], row["reference"], "clickout item"), row["timestamp"]
            ),
            get_stats_func=lambda acc, row, item: min(
                row["timestamp"] - acc.get((row["user_id"], item["item_id"], "clickout item"), 0), 1000000
            ),
        ),
        StatsAcc(
            name="last_timestamp_clickout",
            action_types=["clickout item"],
            acc={},
            updater=lambda acc, row: set_key(acc, (row["user_id"], row["impressions_raw"]), row["timestamp"]),
            get_stats_func=lambda acc, row, item: row["timestamp"]
            - acc.get((row["user_id"], item["impressions_raw"]), 0),
        ),
        ClickProbabilityClickOffsetTimeOffset(action_types=["clickout item"]),
        ClickProbabilityClickOffsetTimeOffset(
            name="fake_clickout_prob_time_position_offset",
            action_types=ACTIONS_WITH_ITEM_REFERENCE,
            impressions_type="fake_impressions_raw",
            index_col="fake_index_interacted",
        ),
        SimilarityFeatures(),
        PoiFeatures(),
        IndicesFeatures(
            action_types=["clickout item"], prefix="", impressions_type="impressions_raw", index_key="index_clicked"
        ),
        IndicesFeatures(
            action_types=list(ACTIONS_WITH_ITEM_REFERENCE),
            prefix="fake_",
            impressions_type="fake_impressions_raw",
            index_key="fake_index_interacted",
        ),
        SimilarityFeatures(),
        PriceFeatures(),
        PriceSimilarity(),
    ] + [
        StatsAcc(
            name="{}_count".format(action_type.replace(" ", "_")),
            action_types=[action_type],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], action_type)),
            get_stats_func=lambda acc, row, item: acc.get((row["user_id"], action_type), 0),
        )
        for action_type in ["filter selection"]
    ]

    accs_by_action_type = defaultdict(list)
    for acc in accumulators:
        for action_type in acc.action_types:
            accs_by_action_type[action_type].append(acc)

    return accumulators, accs_by_action_type
