import gzip
import json
from collections import Counter, defaultdict
from copy import deepcopy
from math import log1p, sqrt
from statistics import mean, stdev
from typing import Dict

import joblib
from recsys.data_generator.accumulators_helpers import (
    add_one_nested_key,
    add_to_set,
    append_to_list,
    append_to_list_not_null,
    diff,
    diff_ts,
    increment_key_by_one,
    increment_keys_by_one,
    set_key,
    set_nested_key,
    tryint,
)
from recsys.data_generator.jaccard_sim import ItemPriceSim, JaccardItemSim
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
    This is the base class for the accumulator. All other classes should implement get_stats and update_acc methods.

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


class ItemLastClickoutStatsInSession:
    """
    It measures how many times the item was the last one being clicked
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.last_interaction = {}
        self.last_interaction_counter = defaultdict(int)

    def update_acc(self, row):
        item_id = row["reference"]
        key = (row["user_id"], row["session_id"])
        if key in self.last_interaction:
            old_item_id = self.last_interaction[key]
            self.last_interaction[key] = item_id
            if old_item_id != item_id:
                self.last_interaction_counter[old_item_id] -= 1
                self.last_interaction_counter[item_id] += 1
        else:
            self.last_interaction_counter[item_id] += 1
            self.last_interaction[key] = item_id

    def get_stats(self, row, item):
        output = {}
        output["last_clickout_item_stats"] = self.last_interaction_counter[item["item_id"]]
        return output


class ClickSequenceFeatures:
    """
    Basic information about the sequence of indices users clicked.
    """

    def __init__(self):
        self.current_impression = {}
        self.sequences = defaultdict(list)
        self.action_types = ["clickout item"]

    def update_acc(self, row):
        if row["action_type"] in self.action_types:
            key = (row["user_id"], row["session_id"])
            self.sequences[key].append(row["index_clicked"])

    def get_stats(self, row, item):
        key = (row["user_id"], row["session_id"])
        obs = {}
        sequence = self.sequences[key]

        if sequence:
            obs["click_sequence_min"] = min(sequence)
            obs["click_sequence_max"] = max(sequence)
            obs["click_sequence_min_norm"] = obs["click_sequence_min"] - item["rank"]
            obs["click_sequence_max_norm"] = obs["click_sequence_max"] - item["rank"]
            obs["click_sequence_len"] = len(sequence)
            obs["click_sequence_sd"] = stdev(sequence) if len(sequence) > 1 else 0
            obs["click_sequence_mean"] = mean(sequence)
            obs["click_sequence_mean_norm"] = obs["click_sequence_mean"] - item["rank"]
            obs["click_sequence_gzip_len"], obs["click_sequence_entropy"] = self._seq_entropy(sequence, item["rank"])
        else:
            obs["click_sequence_min"] = -1000
            obs["click_sequence_max"] = -1000
            obs["click_sequence_min_norm"] = -1000
            obs["click_sequence_max_norm"] = -1000
            obs["click_sequence_len"] = -1000
            obs["click_sequence_sd"] = -1000
            obs["click_sequence_mean"] = -1000
            obs["click_sequence_mean_norm"] = -1000
            obs["click_sequence_gzip_len"] = -1000
            obs["click_sequence_entropy"] = -1000
        return obs

    def _seq_entropy(self, sequence, rank):
        seq = ",".join([str(el) for el in sequence]).encode("utf-8")
        seq_with_rank = ",".join([str(el) for el in sequence + [rank]]).encode("utf-8")
        compressed_with_rank = gzip.compress(seq_with_rank)
        compressed_without_rank = gzip.compress(seq)
        return len(compressed_with_rank), (len(compressed_with_rank) / len(compressed_without_rank))


class FakeClickSequenceFeatures:
    """
    Basic information about the sequence of indices users interacted with.
    """

    def __init__(self):
        self.current_impression = {}
        self.sequences = defaultdict(list)
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE

    def update_acc(self, row):
        if row["action_type"] in self.action_types:
            key = (row["user_id"], row["session_id"])
            self.sequences[key].append(row["fake_index_interacted"])

    def get_stats(self, row, item):
        key = (row["user_id"], row["session_id"])
        obs = {}
        sequence = self.sequences[key]

        if sequence:
            obs["fake_click_sequence_min"] = min(sequence)
            obs["fake_click_sequence_max"] = max(sequence)
            obs["fake_click_sequence_min_norm"] = obs["fake_click_sequence_min"] - item["rank"]
            obs["fake_click_sequence_max_norm"] = obs["fake_click_sequence_max"] - item["rank"]
            obs["fake_click_sequence_len"] = len(sequence)
            obs["fake_click_sequence_sd"] = stdev(sequence) if len(sequence) > 1 else 0
            obs["fake_click_sequence_mean"] = mean(sequence)
            obs["fake_click_sequence_mean_norm"] = obs["fake_click_sequence_mean"] - item["rank"]
            obs["fake_click_sequence_gzip_len"], obs["fake_click_sequence_entropy"] = self._seq_entropy(
                sequence, item["rank"]
            )
        else:
            obs["fake_click_sequence_min"] = -1000
            obs["fake_click_sequence_max"] = -1000
            obs["fake_click_sequence_min_norm"] = -1000
            obs["fake_click_sequence_max_norm"] = -1000
            obs["fake_click_sequence_len"] = -1000
            obs["fake_click_sequence_sd"] = -1000
            obs["fake_click_sequence_mean"] = -1000
            obs["fake_click_sequence_mean_norm"] = -1000
            obs["fake_click_sequence_gzip_len"] = -1000
            obs["fake_click_sequence_entropy"] = -1000
        return obs

    def _seq_entropy(self, sequence, rank):
        seq = ",".join([str(el) for el in sequence]).encode("utf-8")
        seq_with_rank = ",".join([str(el) for el in sequence + [rank]]).encode("utf-8")
        compressed_with_rank = gzip.compress(seq_with_rank)
        compressed_without_rank = gzip.compress(seq)
        return len(compressed_with_rank), (len(compressed_with_rank) / len(compressed_without_rank))


class Last10Actions:
    """
    It creates a list of the last 10 actions
    """

    def __init__(self):
        self.current_impression = {}
        self.sequences = defaultdict(list)
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE

    def update_acc(self, row):
        if row["action_type"] in self.action_types:
            key = (row["user_id"], row["session_id"])
            self.sequences[key].append((row["action_type"], row["fake_index_interacted"]))

    def get_stats(self, row, item):
        key = (row["user_id"], row["session_id"])
        obs = {}
        sequence = [(None, None)] * 10 + self.sequences[key]
        for n in range(10):
            action, ind = sequence[-n]
            obs[f"cat_action_index_{n}"] = "{}{}".format(action, ind) if action else ""
            obs[f"cat_action_index_{n}_norm"] = "{}{}".format(action, item["rank"] - ind) if action else ""
        return obs


class ClickProbabilityClickOffsetTimeOffset:
    """
    It creates the feature based on the previous index clicked, current item rank and timestamp difference between
    the previous interaction and current timestamp.
    """

    def __init__(
        self,
        name="clickout_prob_time_position_offset",
        action_types=None,
        impressions_type="impressions_raw",
        index_col="index_clicked",
        probs_path="../../../data/click_probs_by_index.joblib",
    ):
        self.name = name
        self.action_types = action_types
        self.index_col = index_col
        self.impressions_type = impressions_type
        self.probs_path = probs_path
        # tracks the impression per user
        self.current_impression = defaultdict(str)
        self.last_timestamp = {}
        self.last_clickout_position = {}
        self.read_probs()

    def read_probs(self):
        self.probs = joblib.load(self.probs_path)

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
    """
    Features extracted from the Point of Interest (POI).
    What was the last point of interest the user searched for
    What is the CTR of the item when users are searching for this POI.
    """

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
    """
    Features with the last indices the user interacted with.
    It has the last 5 indices and timestamps
    """

    def __init__(
        self, action_types=["clickout item"], impressions_type="impressions_raw", index_key="index_clicked", prefix=""
    ):
        self.action_types = action_types
        self.impressions_type = impressions_type
        self.index_key = index_key
        self.last_indices = defaultdict(list)
        self.last_timestamps = defaultdict(list)
        self.prefix = prefix

    def update_acc(self, row):
        # TODO: reset list when there is a change of sort order?
        if row["action_type"] in self.action_types and row[self.index_key] >= 0:
            self.last_indices[(row["user_id"], row[self.impressions_type])].append(row[self.index_key])
            self.last_timestamps[(row["user_id"], row[self.impressions_type])].append(row["timestamp"])

    def get_stats(self, row, item):
        last_n = 5
        last_indices_raw = self.last_indices[(row["user_id"], row[self.impressions_type])]
        last_indices = [-100] * last_n + last_indices_raw
        last_indices = last_indices[-last_n:]
        diff_last_indices = diff(last_indices + [item["rank"]])

        last_ts_raw = self.last_timestamps[(row["user_id"], row[self.impressions_type])]
        last_ts = [-100] * last_n + last_ts_raw
        last_ts = last_ts[-last_n:]
        diff_last_ts = diff(last_ts + [item["rank"]])

        output = {}
        for n in range(1, last_n + 1):
            if last_indices[-n] != -100:
                output[self.prefix + "last_index_{}".format(n)] = last_indices[-n]
                # output[self.prefix + "last_index_{}_vs_rank".format(n)] = last_indices[-n] - item["rank"]
                output[self.prefix + "last_index_diff_{}".format(n)] = diff_last_indices[-n]
                output[self.prefix + "last_ts_diff_{}".format(n)] = diff_last_ts[-n]
            else:
                output[self.prefix + "last_index_{}".format(n)] = None
                # output[self.prefix + "last_index_{}_vs_rank".format(n)] = None
                output[self.prefix + "last_index_diff_{}".format(n)] = None
                output[self.prefix + "last_ts_diff_{}".format(n)] = None
        n_consecutive = self._calculate_n_consecutive_clicks(last_indices_raw, item["rank"])
        output[self.prefix + "n_consecutive_clicks"] = n_consecutive
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
    """
    Similarity of price of the current item vs items that the user clicked before
    """

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
    """
    Some basic features based on the price
    """

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
    """
    This class calculates similarity measure between the interaction items and the current item.
    The similarities are based on the 3 main dimensions of the item:
    - properties
    - point of interests around the hotel
    - prices
    For each similarity type it calculates:
    - average similarity
    - similarity to the last item
    """

    def __init__(self, type, hashn):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.type = type

        if self.type == "imm":
            self.jacc_sim = JaccardItemSim(path="../../../data/item_metadata_map.joblib")
        elif self.type == "poi":
            self.poi_sim = JaccardItemSim(path="../../../data/item_pois.joblib")
        elif self.type == "price":
            self.price_sim = ItemPriceSim(path="../../../data/item_prices.joblib")
        self.last_item_clickout = defaultdict(int)
        self.user_item_interactions_list = defaultdict(set)
        self.user_item_session_interactions_list = defaultdict(set)
        self.hashn = hashn

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
        if self.type == "imm":
            if self.hashn == 0:
                output["item_similarity_to_last_clicked_item"] = self.jacc_sim.two_items(
                    last_item_clickout, item["item_id"]
                )
            elif self.hashn == 1:
                output["avg_similarity_to_interacted_items"] = self.jacc_sim.list_to_item(
                    user_item_interactions_list, item_id
                )
            elif self.hashn == 2:
                output["avg_similarity_to_interacted_session_items"] = self.jacc_sim.list_to_item(
                    user_item_session_interactions_list, item_id
                )
        elif self.type == "price":
            if self.hashn == 0:
                output["avg_price_similarity_to_interacted_items"] = self.price_sim.list_to_item(
                    user_item_interactions_list, item_id
                )
            elif self.hashn == 1:
                output["avg_price_similarity_to_interacted_session_items"] = self.price_sim.list_to_item(
                    user_item_session_interactions_list, item_id
                )
        elif self.type == "poi":
            if self.hashn == 0:
                output["poi_item_similarity_to_last_clicked_item"] = self.poi_sim.two_items(last_item_clickout, item_id)
            elif self.hashn == 1:
                output["poi_avg_similarity_to_interacted_items"] = self.poi_sim.list_to_item(
                    user_item_interactions_list, item_id
                )
            elif self.hashn == 2:
                output["num_pois"] = len(self.poi_sim.imm[item_id])
        return output


class ItemCTR:
    """
    Basic class that calculates the CTR for the item.
    The features are:
    - number of clicks
    - number of impressions
    - CTR
    - CTR corrected (it includes only the impressions that were below "above" the item)
    """

    def __init__(self, action_types):
        self.action_types = action_types
        self.clicks = defaultdict(int)
        self.impressions = defaultdict(int)
        self.impressions_corr = defaultdict(int)

    def update_acc(self, row):
        if not row["reference"].isnumeric():
            return
        self.clicks[row["reference"]] += 1
        for rank, item_id in enumerate(row["impressions"]):
            self.impressions[item_id] += 1
            if rank <= row["index_clicked"]:
                self.impressions_corr[item_id] += 1

    def get_stats(self, row, item):
        output = {}
        output["clickout_item_clicks"] = self.clicks[item["item_id"]]
        output["clickout_item_impressions"] = self.impressions[item["item_id"]]
        output["clickout_item_ctr"] = output["clickout_item_clicks"] / (output["clickout_item_impressions"] + 1)
        output["clickout_item_ctr_corr"] = output["clickout_item_clicks"] / (self.impressions_corr[item["item_id"]] + 1)
        return output


class ItemCTRInteractions:
    """
    Similar to ItemCTR but it is based on the interactions.
    """

    def __init__(self):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.clicks = defaultdict(int)
        self.impressions = defaultdict(int)
        self.impressions_corr = defaultdict(int)

    def update_acc(self, row):
        if not row["reference"].isnumeric():
            return
        self.clicks[row["reference"]] += 1
        for rank, item_id in enumerate(row["fake_impressions"]):
            self.impressions[item_id] += 1

    def get_stats(self, row, item):
        output = {}
        output["interact_item_clicks"] = self.clicks[item["item_id"]]
        output["interact_item_impressions"] = self.impressions[item["item_id"]]
        output["interact_item_ctr"] = output["interact_item_clicks"] / (output["interact_item_impressions"] + 1)
        return output


class ItemAverageRank:
    """
    It calculate the average, the last rank per each item
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.ranks = defaultdict(list)

    def update_acc(self, row):
        if not row["reference"].isnumeric():
            return
        key = (row["user_id"], row["session_id"], int(row["reference"]))
        self.ranks[key].append(row["index_clicked"])

    def get_stats(self, row, item):
        obs = {}
        key = (row["user_id"], row["session_id"], int(item["item_id"]))
        obs["item_last_rank"] = self.ranks[key][-1] if self.ranks[key] else -1
        obs["item_avg_rank"] = sum(self.ranks[key]) / (len(self.ranks[key]) + 1)
        return obs


class ItemCTRRankWeighted:
    """
    CTR weighted by the ranking.
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.clicks = defaultdict(int)
        self.impressions = defaultdict(int)

    def update_acc(self, row):
        if row["index_clicked"] != -1000:
            self.clicks[row["reference"]] += row["index_clicked"] + 1
            for ind, item_id in enumerate(row["impressions"]):
                self.impressions[item_id] += ind + 1

    def get_stats(self, row, item):
        output = {}
        output["clickout_item_clicks_rank_weighted"] = self.clicks[item["item_id"]]
        output["clickout_item_impressions_rank_weighted"] = self.impressions[item["item_id"]]
        output["clickout_item_ctr_rank_weighted"] = output["clickout_item_clicks_rank_weighted"] / (
            output["clickout_item_impressions_rank_weighted"] + 1
        )
        return output


class UserItemAttentionSpan:
    """
    This class calculates how much time the user spends when he interacted with the item.
    """

    def __init__(self):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.user_interaction_times = defaultdict(list)
        self.user_last_interaction_item = {}
        self.user_last_interaction_ts = {}

    def update_acc(self, row):
        key = (row["user_id"], row["session_id"])
        new_item_id = row["reference"]
        new_ts = row["timestamp"]
        old_item_id = self.user_last_interaction_item.get(key)
        old_ts = self.user_last_interaction_ts.get(key)
        if new_item_id != old_item_id and new_item_id and old_item_id:
            # some other item had interaction
            self.user_interaction_times[key].append(new_ts - old_ts)
        self.user_last_interaction_item[key] = new_item_id
        self.user_last_interaction_ts[key] = new_ts

    def get_stats(self, row, item):
        key = (row["user_id"], row["session_id"])
        new_ts = row["timestamp"]
        old_ts = self.user_last_interaction_ts.get(key, 0)
        obs = {}
        if key in self.user_interaction_times:
            obs["user_item_avg_attention"] = sum(self.user_interaction_times[key]) / (
                len(self.user_interaction_times[key])
            )
            obs["is_item_within_avg_span"] = int(
                ((new_ts - old_ts) < obs["user_item_avg_attention"])
                and (self.user_last_interaction_item[key] == item["item_id"])
            )
            obs["is_item_within_avg_span_2s"] = int(
                ((new_ts - old_ts) < (2 * obs["user_item_avg_attention"]))
                and (self.user_last_interaction_item[key] == item["item_id"])
            )
        else:
            obs["user_item_avg_attention"] = -1
            obs["is_item_within_avg_span"] = -1
            obs["is_item_within_avg_span_2s"] = -1
        return obs


class GlobalClickoutTimestamp:
    """
    This class extracts the timestamp of the last click on the item (globally).
    It also checks if the click was from the same user.
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.last_ts = {}
        self.last_user = {}

    def update_acc(self, row):
        if not row["reference"].isnumeric():
            return
        self.last_ts[int(row["reference"])] = row["timestamp"]
        self.last_user[int(row["reference"])] = row["user_id"]

    def get_stats(self, row, item):
        obs = {}
        if self.last_ts.get(int(item["item_id"])):
            obs["last_item_timestamp"] = row["timestamp"] - self.last_ts.get(int(item["item_id"]))
        else:
            obs["last_item_timestamp"] = None
        obs["last_item_click_same_user"] = int(self.last_user.get(int(item["item_id"]), None) == row["user_id"])
        return obs


class DistinctInteractions:
    """
    This class is quite interesting. We noticed that the users have some tendencies to interact with unique
    items or the same items as before. So if the user interacts with the items that are mostly unique that means
    that the next item will probably also be unique.
    """

    def __init__(self, name, action_types, by="timestamp"):
        self.name = name
        self.action_types = action_types
        self.counter = defaultdict(int)
        self.item_set = defaultdict(set)

    def update_acc(self, row):
        key = row["user_id"]
        if row["reference"].isnumeric():
            self.item_set[key].add(int(row["reference"]))
            self.counter[key] += 1

    def get_stats(self, row, item):
        key = row["user_id"]
        uniq = len(self.item_set[key])
        all = self.counter[key]

        """
        If the interactions are unique so for example there were 10 interactions with 10 items
        then the probability of another unique interaction is very high (rule of succession = 11/12).
        
        uniq interactions = 10
        all interactions = 10
        
        item  was_interaction  prob
        A     true             1-11/12 = 0.08333
        B     true             1-11/12 = 0.08333       
        C     false            11/12   = 0.91666
        D     false            11/12   = 0.91666
        """

        obs = {}
        obs[f"{self.name}_uniq_interactions"] = uniq / (all + 1)
        if int(item["item_id"]) in self.item_set[key]:
            obs[f"{self.name}_item_uniq_prob"] = 1 - ((uniq + 1) / (all + 2))
        else:
            obs[f"{self.name}_item_uniq_prob"] = (uniq + 1) / (all + 2)

        return obs


def fit_lr(X, Y):
    def mean(Xs):
        return sum(Xs) / len(Xs)

    m_X = mean(X)
    m_Y = mean(Y)

    def std(Xs, m):
        normalizer = len(Xs) - 1
        return sqrt(sum((pow(x - m, 2) for x in Xs)) / normalizer)

    # assert np.round(Series(X).std(), 6) == np.round(std(X, m_X), 6)

    def pearson_r(Xs, Ys):

        sum_xy = 0
        sum_sq_v_x = 0
        sum_sq_v_y = 0

        for (x, y) in zip(Xs, Ys):
            var_x = x - m_X
            var_y = y - m_Y
            sum_xy += var_x * var_y
            sum_sq_v_x += pow(var_x, 2)
            sum_sq_v_y += pow(var_y, 2)
        return sum_xy / sqrt(sum_sq_v_x * sum_sq_v_y)

    # assert np.round(Series(X).corr(Series(Y)), 6) == np.round(pearson_r(X, Y), 6)

    r = pearson_r(X, Y)

    b = r * (std(Y, m_Y) / std(X, m_X))
    A = m_Y - b * m_X

    def line(x):
        return b * x + A

    return line


class ClickSequenceTrend:
    """
    We calculate the click trend of the user. If there is a sequence the user follows it should be more or less
    reflected in the simple model.

    We use a linear regression and the slope of minimum and maximum index.
    """

    def __init__(self, method="minmax", by="user_id"):
        self.by = by
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.user_ind = defaultdict(list)
        self.method = method

    def update_acc(self, row):
        if row["fake_index_interacted"] == -1000:
            return
        self.user_ind[row[self.by]].append((row["fake_index_interacted"], row["timestamp"]))

    def get_stats(self, row, item):
        obs = {}
        obs[f"predicted_ind_{self.method}_by_{self.by}"] = -1
        obs[f"predicted_ind_rel_{self.method}_by_{self.by}"] = -1
        obs[f"ind_per_ts_{self.method}_by_{self.by}"] = -1
        if len(self.user_ind[row[self.by]]) >= 2:
            max_ind, max_ts = self.user_ind[row[self.by]][-1]
            if self.method == "minmax":
                min_ind, min_ts = self.user_ind[row[self.by]][0]
                if max_ts - min_ts > 0:
                    ind_per_ts = (max_ind - min_ind) / (max_ts - min_ts)
                    ts_passed = row["timestamp"] - max_ts
                    obs[f"ind_per_ts_{self.method}_by_{self.by}"] = ind_per_ts
                    obs[f"predicted_ind_{self.method}_by_{self.by}"] = max_ind + ts_passed * ind_per_ts
            elif self.method == "lr":
                X = [row["timestamp"] - ts for ind, ts in self.user_ind[row[self.by]]]
                Y = [ind for ind, ts in self.user_ind[row[self.by]]]
                try:
                    line = fit_lr(X, Y)
                    obs[f"predicted_ind_{self.method}_by_{self.by}"] = line(row["timestamp"] - max_ts)
                except ZeroDivisionError:
                    obs[f"predicted_ind_{self.method}_by_{self.by}"] = -1
            obs[f"predicted_ind_rel_{self.method}_by_{self.by}"] = (
                obs[f"predicted_ind_{self.method}_by_{self.by}"] - item["rank"]
            )
        return obs


class SimilarUsersItemInteraction:
    """
    This is an accumulator that given interaction with items
    Finds users who interacted with the same items and then gathers statistics of interaction
    from them.
    In other words if the users and items are a bipartite graph this class calculates two passes over the edges 
    of this graph.
    
    ITEM <--- interacts with --- USER --- interacts with ---> ITEM
    """

    def __init__(self):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.items_users = defaultdict(set)
        self.users_items = defaultdict(set)
        self.cache_key = None
        self.item_stats_cached = None

    def update_acc(self, row):
        if row["is_test"] == "0":
            self.items_users[row["reference"]].add(row["user_id"])
            self.users_items[row["user_id"]].add(row["reference"])

    def get_stats(self, row, item):
        items_stats = self.read_stats_from_cache(row)
        obs = {}
        obs["similar_users_item_interaction"] = items_stats[item["item_id"]]
        return obs

    def read_stats_from_cache(self, row):
        key = (row["user_id"], row["timestamp"])
        if self.cache_key == key:
            items_stats = self.item_stats_cached
        else:
            items_stats = self.get_items_stats(row)
            self.item_stats_cached = items_stats
            self.cache_key = key
        return items_stats

    def get_items_stats(self, row):
        items = defaultdict(int)
        for item_id in self.users_items[row["user_id"]]:
            for user_id in self.items_users[item_id]:
                # discard the self similarity
                if user_id == row["user_id"]:
                    continue
                for item_id_2 in self.users_items[user_id]:
                    items[item_id_2] += 1
        return items


class GlobalTimestampPerItem:
    """
    Similar to the previous class which extracts the timestamps of clicks on the item
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.timestamp = {}
        self.last_user = {}

    def update_acc(self, row):
        self.timestamp[row["reference"]] = row["timestamp"]
        self.last_user[row["reference"]] = row["user_id"]

    def get_stats(self, row, item):
        output = {}
        output["last_item_time_diff_same_user"] = None
        output["last_item_last_user_id"] = None
        output["last_item_time_diff"] = None

        if item["item_id"] in self.timestamp:
            output["last_item_last_user_id"] = self.last_user[item["item_id"]]
            output["last_item_time_diff"] = row["timestamp"] - self.timestamp[item["item_id"]]
            output["last_item_time_diff_same_user"] = output["last_item_time_diff"]
            if row["user_id"] == self.last_user[item["item_id"]]:
                output["last_item_time_diff_same_user"] = None
        return output


class TimeSinceSessionStart:
    """
    Calculates the time since the start of the session
    """

    def __init__(self):
        self.action_types = ALL_ACTIONS
        self.session_start = {}

    def update_acc(self, row):
        key = (row["user_id"], row["session_id"])
        if key not in self.session_start:
            self.session_start[key] = row["timestamp"]

    def get_stats(self, row, item):
        key = (row["user_id"], row["session_id"])
        obs = {}
        if key in self.session_start:
            obs["session_start_ts"] = row["timestamp"] - self.session_start[key]
        else:
            obs["session_start_ts"] = 0
        return obs


class NumberOfSessions:
    """
    Calculates the number of sessions of the current user
    """

    def __init__(self):
        self.action_types = ALL_ACTIONS
        self.session_count = defaultdict(set)

    def update_acc(self, row):
        self.session_count[row["user_id"]].add(row["session_id"])

    def get_stats(self, row, item):
        obs = {}
        obs["session_count"] = len(self.session_count[row["user_id"]])
        return obs


class TimeSinceUserStart:
    """
    Calculates time since first user action
    """

    def __init__(self):
        self.action_types = ALL_ACTIONS
        self.start = {}

    def update_acc(self, row):
        key = row["user_id"]
        if key not in self.start:
            self.start[key] = row["timestamp"]

    def get_stats(self, row, item):
        key = row["user_id"]
        obs = {}
        if key in self.start:
            obs["user_start_ts"] = row["timestamp"] - self.start[key]
        else:
            obs["user_start_ts"] = 0
        return obs


class AllFilters:
    """
    Extracts all the filters the user used throughout the history
    """

    def __init__(self):
        self.action_types = ["filter selection"]
        self.filters_by_user = defaultdict(set)

    def update_acc(self, row):
        self.filters_by_user[row["user_id"]].add(row["reference"])

    def get_stats(self, row, item):
        obs = {}
        obs["alltime_filters"] = "|".join(self.filters_by_user[row["user_id"]])
        return obs


class MostSimilarUserItemInteraction:
    """
    This is an accumulator that given interaction with items
    Finds users who interacted with the same items and then gathers statistics of interaction
    from them
    
    This class is similar to SimilarUsersItemInteraction but it only focuses on the most similar users.
    """

    def __init__(self):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.items_users = defaultdict(set)
        self.users_items = defaultdict(set)
        self.cache_key = None
        self.item_stats_cached = None

    def update_acc(self, row):
        if row["is_test"] == "0":
            self.items_users[row["reference"]].add(row["user_id"])
            self.users_items[row["user_id"]].add(row["reference"])

    def get_stats(self, row, item):
        items_stats = self.read_stats_from_cache(row)
        obs = {}
        obs["most_similar_item_interaction"] = items_stats[item["item_id"]]
        return obs

    def read_stats_from_cache(self, row):
        key = (row["user_id"], row["timestamp"])
        if self.cache_key == key:
            items_stats = self.item_stats_cached
        else:
            items_stats = self.get_items_stats(row)
            self.item_stats_cached = items_stats
            self.cache_key = key
        return items_stats

    def get_items_stats(self, row):
        this_user_items = self.users_items[row["user_id"]]
        best_user_id = None
        best_intersection_len = 0
        for item_id in this_user_items:
            for other_user_id in self.items_users[item_id]:
                if other_user_id == row["user_id"]:
                    continue
                intersection_len = len(self.users_items[other_user_id] | this_user_items)
                if intersection_len > best_intersection_len:
                    best_user_id = other_user_id
                    best_intersection_len = intersection_len
        items = defaultdict(int)
        for item_id in self.users_items[best_user_id]:
            items[item_id] = 1
        return items


class MostSimilarUserItemInteractionv2:
    """
    This is an accumulator that given interaction with items
    Finds users who interacted with the same items and then gathers statistics of interaction
    from them
    """

    def __init__(self, k=1):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.items_users = defaultdict(set)
        self.users_items = defaultdict(set)
        self.k = k
        self.cache_key = None
        self.item_stats_cached = None

    def update_acc(self, row):
        self.items_users[row["reference"]].add(row["user_id"])
        self.users_items[row["user_id"]].add(row["reference"])

    def get_stats(self, row, item):
        items_stats = self.read_stats_from_cache(row)
        obs = {}
        obs["most_similar_item_interaction_k_{}".format(self.k)] = items_stats[item["item_id"]]
        return obs

    def read_stats_from_cache(self, row):
        key = (row["user_id"], row["timestamp"])
        if self.cache_key == key:
            items_stats = self.item_stats_cached
        else:
            items_stats = self.get_items_stats(row)
            self.item_stats_cached = items_stats
            self.cache_key = key
        return items_stats

    def get_items_stats(self, row):
        this_user_items = self.users_items[row["user_id"]]
        user_stats = []
        for item_id in this_user_items:
            for other_user_id in self.items_users[item_id]:
                if other_user_id == row["user_id"]:
                    continue
                intersection_len = len(self.users_items[other_user_id] | this_user_items)
                user_stats.append((other_user_id, intersection_len))
        selected_users = sorted(user_stats, key=lambda x: x[1], reverse=True)[: self.k]
        items = defaultdict(int)
        for user_id, _ in selected_users:
            for item_id in self.users_items[user_id]:
                items[item_id] = 1
        return items


class ItemCTRInSequence:
    """
    Calculates statistics of items which were clicked as the last in sequence.
    Because the competition focuses on the last clickout actions we wanted to measure
    how likely the item was clicked as the last one in the sequence.
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.item_clicks_when_last = defaultdict(int)
        self.item_impressions_when_last = defaultdict(int)
        self.item_click_in_rev_seq = defaultdict(int)
        self.item_count_in_rev_seq = defaultdict(int)

    def update_acc(self, row):
        try:
            item_id = int(row["reference"])
        except:
            return
        if int(row["clickout_step_rev"]) == 1:
            self.item_clicks_when_last[item_id] += 1
            for item_id_imp in row["impressions"]:
                item_id_imp = int(item_id_imp)
                self.item_impressions_when_last[item_id_imp] += 1
        self.item_click_in_rev_seq[item_id] += int(row["clickout_step_rev"])
        self.item_count_in_rev_seq[item_id] += 1

    def get_stats(self, row, item):
        item_id = int(item["item_id"])
        obs = {}
        obs["item_clicks_when_last"] = self.item_clicks_when_last[item_id]
        obs["item_impressions_when_last"] = self.item_impressions_when_last[item_id]
        obs["item_ctr_when_last"] = obs["item_clicks_when_last"] / (obs["item_impressions_when_last"] + 1)
        obs["item_average_seq_pos"] = self.item_click_in_rev_seq[item_id] / (self.item_count_in_rev_seq[item_id] + 1)
        return obs


class PriceSorted:
    """
    Sometimes the filter says that the prices are sorted but they are not.
    This class creates a feature `wrong_price_sorting` if the sort order is not as it should be.
    """

    def __init__(self):
        self.action_types = ["clickout item"]

    def update_acc(self, row):
        pass

    def get_stats(self, row, item):
        prices = row["prices"]
        obs = {}
        obs["price_rem"] = item["price"] % 100
        obs["are_price_sorted"] = int(prices == sorted(prices))
        obs["are_price_sorted_rev"] = int(prices == sorted(prices, reverse=True))

        # calculates the point until the prices are sorted
        obs["prices_sorted_until"] = 0
        for n in range(len(prices)):
            prices_sorted = int(prices == sorted(prices))
            if not prices_sorted:
                break
            obs["prices_sorted_until"] = n

        obs["prices_sorted_until_current_rank"] = int(item["rank"] < n)
        should_be_sorted = int("Sort by Price" in row["current_filters"])
        obs["wrong_price_sorting"] = int(should_be_sorted and not obs["are_price_sorted"])
        return obs


class ActionsTracker:
    """
    This is a very generic class that extracts many features from the events.
    It is probably an overkill and it causes the memory to explode.
    """

    def __init__(self):
        self.action_types = ALL_ACTIONS
        self.all_events_list = defaultdict(lambda: defaultdict(list))
        self.int_events_list = defaultdict(list)
        self.max_timestamp = defaultdict(int)

    def update_acc(self, row: Dict):
        if self.max_timestamp[row["user_id"]] == 0:
            self.max_timestamp[row["user_id"]] = row["timestamp"]

        # aggregate to lists
        new_row = self.prepare_new_row(row)

        self.all_events_list[row["user_id"]][row["action_type"]].append(new_row)
        if row["action_type"] in ACTIONS_WITH_ITEM_REFERENCE:
            self.int_events_list[row["user_id"]].append(new_row)

    def prepare_new_row(self, row):
        new_row = {}
        new_row["action_type"] = row["action_type"]
        new_row["timestamp"] = row["timestamp"]
        new_row["reference"] = row["reference"]
        new_row["fake_prices"] = row["fake_prices"]
        new_row["fake_impressions"] = row["fake_impressions"]
        return new_row

    def get_stats(self, row, item):
        all_events_list = self.all_events_list[row["user_id"]]
        max_timestamp = row["timestamp"]
        obs = {}
        for action_type in all_events_list.keys():
            for event_num, new_row in enumerate(all_events_list[action_type][::-1][:10]):
                impressions = new_row["fake_impressions"]
                prices = new_row["fake_prices"].split("|")
                # import ipdb; ipdb.set_trace()
                if action_type == "clickout item" and event_num <= 1:
                    for rank, (item_id, price) in enumerate(zip(impressions, prices)):
                        price = int(price)
                        obs[f"co_price_{rank:02d}_{event_num:02d}"] = log1p(price)

                obs[f"{action_type}_{event_num:02d}_timestamp"] = log1p(max_timestamp - new_row["timestamp"])
                if new_row["action_type"] in ACTIONS_WITH_ITEM_REFERENCE:
                    impressions = new_row["fake_impressions"]
                    if new_row["reference"] in impressions:
                        obs[f"{action_type}_rank_{event_num:02d}"] = impressions.index(new_row["reference"]) + 1
                        obs[f"{action_type}_rank_{event_num:02d}_rel"] = item["rank"] - impressions.index(
                            new_row["reference"]
                        )

        int_events_list = self.int_events_list[row["user_id"]]
        for event_num, new_row in enumerate(int_events_list[::-1][:10]):
            obs[f"interaction_{event_num:02d}_timestamp"] = log1p(max_timestamp - new_row["timestamp"])
            impressions = new_row["fake_impressions"]
            if new_row["reference"] in impressions:
                obs[f"interaction_rank_{event_num:02d}"] = impressions.index(new_row["reference"]) + 1
                obs[f"interaction_rank_{event_num:02d}_rel"] = item["rank"] - impressions.index(new_row["reference"])

        return {"actions_tracker": json.dumps(obs)}


class PairwiseCTR:
    """
    This class calculates statistics of items pairs. If the items A and B are often next to each other it can
    be that sometimes A wins more often (as it should be because it is higher in the ranking) or B wins more
    often which can mean that the ranking is wrong.
    """

    LEFT_WON = -1
    RIGHT_WON = 1
    DRAW = 0

    def __init__(self):
        self.action_types = ["clickout item"]
        self.pairs = defaultdict(Counter)

    def update_acc(self, row: Dict):
        if not row["reference"].isnumeric():
            return
        impressions = list(map(int, row["impressions"]))
        if not row["reference"].isnumeric():
            return
        ref = int(row["reference"])
        for bigr in self.zipngram3(impressions, 2):
            l, r = bigr
            if ref == l:
                self.pairs[bigr].update([self.LEFT_WON])
            elif ref == r:
                self.pairs[bigr].update([self.RIGHT_WON])
            else:
                self.pairs[bigr].update([self.DRAW])

    def get_stats(self, row, item):
        impressions = list(map(int, row["impressions"]))
        position = item["rank"]

        try:
            prv_item = impressions[position - 1]
        except IndexError:
            prv_item = None
        this_item = impressions[position]
        try:
            next_item = impressions[position + 1]
        except IndexError:
            next_item = None

        obs = {}
        obs["pairwise_1_ctr_left_won"] = self.pairs[(prv_item, this_item)][self.LEFT_WON]
        obs["pairwise_1_ctr_right_won"] = self.pairs[(prv_item, this_item)][self.RIGHT_WON]
        obs["pairwise_1_ctr_draw"] = self.pairs[(prv_item, this_item)][self.DRAW]
        obs["pairwise_1_rel"] = obs["pairwise_1_ctr_left_won"] / (obs["pairwise_1_ctr_right_won"] + 1)
        obs["pairwise_2_ctr_left_won"] = self.pairs[(this_item, next_item)][self.LEFT_WON]
        obs["pairwise_2_ctr_right_won"] = self.pairs[(this_item, next_item)][self.RIGHT_WON]
        obs["pairwise_2_ctr_draw"] = self.pairs[(this_item, next_item)][self.DRAW]
        obs["pairwise_2_rel"] = obs["pairwise_2_ctr_left_won"] / (obs["pairwise_2_ctr_right_won"] + 1)
        return obs

    def zipngram3(self, words, n=2):
        return zip(*[words[i:] for i in range(n)])


class RankOfItemsFreshClickout:
    """
    This class calculates rank of items that were clicked in the first 2 steps of the user history.
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.positions = defaultdict(Counter)

    def update_acc(self, row: Dict):
        if int(row["step"]) <= 2:
            impressions = list(map(int, row["impressions"]))
            for rank, item_id in enumerate(impressions):
                self.positions[item_id].update([rank])

    def get_stats(self, row, item):
        obs = {}
        s = sum([cnt for cnt in self.positions[int(item["item_id"])].values()])
        avg = sum([rank * cnt for rank, cnt in self.positions[int(item["item_id"])].items()]) / (s + 1)
        obs["average_fresh_rank"] = avg
        obs["average_fresh_rank_rel"] = avg - item["rank"]
        return obs


class SequenceClickout:
    """
    Was the item in the previous impression.
    Similarity between current impression and the previous one
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.last_impressions = {}
        self.last_prices = {}

    def update_acc(self, row: Dict):
        self.last_impressions[row["user_id"]] = set(list(map(int, row["impressions"])))

    def get_stats(self, row, item):
        impressions = set(list(map(int, row["impressions"])))
        obs = {}
        last_impressions = self.last_impressions.get(row["user_id"], set())
        obs["item_was_in_prv_clickout"] = int(int(item["item_id"]) in last_impressions)
        obs["item_clickouts_intersection"] = len(last_impressions & impressions)
        return obs


class SameImpressionsDifferentUser:
    """
    If there was the same impression with the different user calculate CTR of the items
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        # impressions -> (user, item)
        self.impressions_clicks = defaultdict(list)

    def update_acc(self, row: Dict):
        if row["reference"].isnumeric():
            key = (row["user_id"], int(row["reference"]))
            self.impressions_clicks[row["impressions_raw"]].append(key)

    def get_stats(self, row, item):
        obs = {}
        key = (row["user_id"], int(item["item_id"]))
        all_clicks = [it for user, it in self.impressions_clicks[row["impressions_raw"]] if user != row["user_id"]]
        item_clicks = [it for it in all_clicks if it == int(item["item_id"])]
        obs["same_impression_different_user_clicks"] = len(item_clicks)
        obs["same_impression_different_user_ctr"] = len(item_clicks) / (len(all_clicks) + 1)
        return obs


class SameImpressionsDifferentUserTopN:
    """
    Same as SameImpressionsDifferentUser but only take into account the top N impressions
    when calculating the "exactness" of the impressions
    """

    def __init__(self, topn=5):
        self.topn = topn
        self.action_types = ["clickout item"]
        # impressions -> (user, item)
        self.impressions_clicks = defaultdict(list)

    def update_acc(self, row: Dict):
        if row["reference"].isnumeric():
            impressions_str = self.extract_top_impressions(row)
            key = (row["user_id"], int(row["reference"]))
            self.impressions_clicks[impressions_str].append(key)

    def extract_top_impressions(self, row):
        return "|".join(row["impressions_raw"].split("|")[: self.topn])

    def get_stats(self, row, item):
        obs = {}
        key = (row["user_id"], int(item["item_id"]))
        impressions_str = self.extract_top_impressions(row)
        all_clicks = [it for user, it in self.impressions_clicks[impressions_str] if user != row["user_id"]]
        item_clicks = [it for it in all_clicks if it == int(item["item_id"])]
        obs[f"same_impression_different_user_clicks_{self.topn}"] = len(item_clicks)
        obs[f"same_impression_different_user_ctr_{self.topn}"] = len(item_clicks) / (len(all_clicks) + 1)
        return obs


class SameFakeImpressionsDifferentUser:
    """
    This is the same as SameImpressionsDifferentUser but we use all interactions to calculate the CTR
    """

    def __init__(self):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        # impressions -> (user, item)
        self.impressions_clicks = defaultdict(list)

    def update_acc(self, row: Dict):
        if row["reference"].isnumeric():
            key = (row["user_id"], int(row["reference"]))
            self.impressions_clicks[row["fake_impressions_raw"]].append(key)

    def get_stats(self, row, item):
        obs = {}
        key = (row["user_id"], int(item["item_id"]))
        all_clicks = [it for user, it in self.impressions_clicks[row["impressions_raw"]] if user != row["user_id"]]
        item_clicks = [it for it in all_clicks if it == int(item["item_id"])]
        obs["same_fake_impression_different_user_clicks"] = len(item_clicks)
        obs["same_fake_impression_different_user_ctr"] = len(item_clicks) / (len(all_clicks) + 1)
        return obs


class RankBasedCTR:
    """
    This class calculates the CTR of the item combined with the ranking of the item.
    When extracting the statistics we use a smooth version of CTR so take the CTR at the position with the
    highest weight and surrounding ones with lower weights.
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.item_rank_clicks = defaultdict(lambda: dict(zip(range(25), [0] * 25)))
        self.item_rank_impressions = defaultdict(lambda: dict(zip(range(25), [0] * 25)))

    def update_acc(self, row: Dict):
        if not row["reference"].isnumeric():
            return
        item_id = int(row["reference"])
        if row["index_clicked"] == -1000:
            return
        impressions = list(map(int, row["impressions"]))
        for rank, item_id in enumerate(impressions):
            self.item_rank_impressions[item_id][rank] += 1
        self.item_rank_clicks[item_id][row["index_clicked"]] += 1

    def get_stats(self, row, item):
        item_id = int(item["item_id"])
        rank = int(item["rank"])
        obs = {}
        if rank == 0:
            obs["rank_based_ctr"] = (
                (self.item_rank_clicks[item_id][0] + 1) / (self.item_rank_impressions[item_id][0] + 2) * 0.5
                + (self.item_rank_clicks[item_id][1] + 1) / (self.item_rank_impressions[item_id][1] + 2) * 0.3
                + (self.item_rank_clicks[item_id][2] + 1) / (self.item_rank_impressions[item_id][2] + 2) * 0.2
            )
        elif rank == 24:
            obs["rank_based_ctr"] = (
                (self.item_rank_clicks[item_id][24] + 1) / (self.item_rank_impressions[item_id][24] + 2) * 0.5
                + (self.item_rank_clicks[item_id][23] + 1) / (self.item_rank_impressions[item_id][23] + 2) * 0.3
                + (self.item_rank_clicks[item_id][22] + 1) / (self.item_rank_impressions[item_id][22] + 2) * 0.2
            )
        else:
            obs["rank_based_ctr"] = (
                (self.item_rank_clicks[item_id][rank - 1] + 1)
                / (self.item_rank_impressions[item_id][rank - 1] + 2)
                * 0.25
                + (self.item_rank_clicks[item_id][rank] + 1) / (self.item_rank_impressions[item_id][rank] + 2) * 0.5
                + (self.item_rank_clicks[item_id][rank + 1] + 1)
                / (self.item_rank_impressions[item_id][rank + 1] + 2)
                * 0.25
            )
        return obs


class AccByKey:
    """
    This is a meta accumulator that wraps other accumulators.
    It can calculate exactly the same statistics as the base accumulator but it works within some group.
    So for example we can calculate ItemCTR per device like this

    AccByKey(ItemCTR(), key="device")
    """

    def __init__(self, base_acc, key):
        self.key = key
        self.base_acc = base_acc
        self.action_types = base_acc.action_types
        self.accs_by_key = {}

    def update_acc(self, row: Dict):
        row["platform_device"] = row["platform"] + row["device"]
        if row[self.key] not in self.accs_by_key:
            self.accs_by_key[row[self.key]] = deepcopy(self.base_acc)
        self.accs_by_key[row[self.key]].update_acc(row)
        del row["platform_device"]

    def get_stats(self, row, item):
        row["platform_device"] = row["platform"] + row["device"]
        if row[self.key] not in self.accs_by_key:
            obs = self.base_acc.get_stats(row, item)
        else:
            obs = self.accs_by_key[row[self.key]].get_stats(row, item)
        for k in list(obs):
            obs[f"{k}_by_{self.key}"] = obs[k]
            del obs[k]
        del row["platform_device"]
        return obs


def group_accumulators(accumulators):
    accs_by_action_type = defaultdict(list)
    for acc in accumulators:
        for action_type in acc.action_types:
            accs_by_action_type[action_type].append(acc)
    return accs_by_action_type


def get_accumulators(hashn=None):
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
            name="last_item_clickout",
            action_types=["clickout item"],
            acc={},
            updater=lambda acc, row: set_key(acc, row["user_id"], row["reference"]),
            get_stats_func=lambda acc, row, item: acc.get(row["user_id"], 0),
        ),
        # item ctr
        ItemCTR(action_types=["clickout item"]),
        AccByKey(ItemCTR(action_types=["clickout item"]), key="platform_device"),
        AccByKey(ItemCTR(action_types=["clickout item"]), key="platform"),
        AccByKey(ItemCTR(action_types=["clickout item"]), key="device"),
        ItemCTRInteractions(),
        AccByKey(ItemCTRInteractions(), key="platform_device"),
        AccByKey(ItemCTRInteractions(), key="platform"),
        AccByKey(ItemCTRInteractions(), key="device"),
        StatsAcc(
            name="clickout_item_platform_clicks",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["reference"], row["platform"])),
            get_stats_func=lambda acc, row, item: acc[(item["item_id"], row["platform"])],
        ),
        StatsAcc(
            name="clickout_user_item_clicks",
            action_types=["clickout item"],
            acc=defaultdict(int),
            updater=lambda acc, row: increment_key_by_one(acc, (row["user_id"], row["reference"])),
            get_stats_func=lambda acc, row, item: acc[(row["user_id"], item["item_id"])],
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
                row["timestamp"] - acc.get((row["user_id"], item["item_id"], "interaction item image"), 0), 1_000_000
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
                row["timestamp"] - acc.get((row["user_id"], item["item_id"], "clickout item"), 0), 1_000_000
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
        SimilarityFeatures("imm", hashn=0),
        SimilarityFeatures("imm", hashn=1),
        SimilarityFeatures("imm", hashn=2),
        SimilarityFeatures("poi", hashn=0),
        SimilarityFeatures("poi", hashn=1),
        SimilarityFeatures("poi", hashn=2),
        SimilarityFeatures("price", hashn=0),
        SimilarityFeatures("price", hashn=1),
        PoiFeatures(),
        ItemLastClickoutStatsInSession(),
        # ItemAttentionSpan(),
        IndicesFeatures(
            action_types=["clickout item"], prefix="", impressions_type="impressions_raw", index_key="index_clicked"
        ),
        IndicesFeatures(
            action_types=list(ACTIONS_WITH_ITEM_REFERENCE),
            prefix="fake_",
            impressions_type="fake_impressions_raw",
            index_key="fake_index_interacted",
        ),
        PriceFeatures(),
        PriceSimilarity(),
        SimilarUsersItemInteraction(),
        MostSimilarUserItemInteraction(),
        GlobalTimestampPerItem(),
        ClickSequenceFeatures(),
        FakeClickSequenceFeatures(),
        TimeSinceSessionStart(),
        TimeSinceUserStart(),
        NumberOfSessions(),
        AllFilters(),
        ItemCTRInSequence(),
        ItemCTRRankWeighted(),
        Last10Actions(),
        PriceSorted(),
        ActionsTracker(),
        DistinctInteractions(name="clickout", action_types=["clickout item"]),
        DistinctInteractions(name="interact", action_types=ACTIONS_WITH_ITEM_REFERENCE),
        PairwiseCTR(),
        RankOfItemsFreshClickout(),
        GlobalClickoutTimestamp(),
        SequenceClickout(),
        RankBasedCTR(),
        ItemAverageRank(),
        UserItemAttentionSpan(),
        SameImpressionsDifferentUser(),
        SameFakeImpressionsDifferentUser(),
        ClickSequenceTrend(method="minmax", by="user_id"),
        ClickSequenceTrend(method="lr", by="user_id"),
        ClickSequenceTrend(method="minmax", by="session_id"),
        ClickSequenceTrend(method="lr", by="session_id"),
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

    if hashn is not None:
        accumulators = [acc for i, acc in enumerate(accumulators) if i % 32 == hashn]
        print("N acc", hashn, len(accumulators))

    return accumulators
