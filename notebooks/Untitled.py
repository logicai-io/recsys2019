# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
import pandas as pd
from recsys.data_generator.generate_training_data import FeatureGenerator
from recsys.data_generator.accumulators import ACTIONS_WITH_ITEM_REFERENCE
from recsys.metric import mrr_fast
from collections import defaultdict
from tqdm import tqdm
import numpy as np


# +
class MouseSpeed:
    def __init__(self):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.last_timestamp_per_session = {}
        self.last_index_per_session = {}
        self.mouse_speed = defaultdict(list)

    def update_acc(self, row):
        key = (row["user_id"], row["session_id"])
        if key in self.last_index_per_session:
            if row["timestamp"] > self.last_timestamp_per_session[key] and \
               row["fake_index_interacted"] != self.last_index_per_session[key]:
                time_passed = row["timestamp"] - self.last_timestamp_per_session[key]
                index_diff = abs(row["fake_index_interacted"] - self.last_index_per_session[key])
                self.mouse_speed[row["user_id"]].append(time_passed / index_diff)
        else:
            if row["fake_index_interacted"] != -1000:
                self.last_timestamp_per_session[key] = row["timestamp"]
                self.last_index_per_session[key] = row["fake_index_interacted"]        
    
    def get_stats(self, row, item):
        key = (row["user_id"], row["session_id"])
        output = {}
        output["mouse_speed"] = self._mean(self.mouse_speed[row["user_id"]])
#         if key in self.last_timestamp_per_session:
#             last_timestamp = self.last_timestamp_per_session[key]
#             last_index = self.last_index_per_session[key]
#             if row["timestamp"] < output["mouse_speed"]*item["rank"]*(last_index - row["fake_index_interacted"])
#         output["mouse_speed_rank"] = 
        return output
    
    def _mean(self, values):
        if values:
            return sum(values) / len(values)
        else:
            return 0
        

class SameImpressionUserStats:
    """
    Example definition

    StatsAcc(filter=lambda row: row.action_type == "clickout_item",
             init_acc=defaultdict(int),
             updater=lambda acc, row: acc[(row.user_id, row.item_id)]+=1)
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.view_stats = defaultdict(lambda: defaultdict(int))

    def update_acc(self, row):
        self.view_stats[row["impressions_raw"]][row["user_id"]]
        self.updater(self.acc, row)

    def get_stats(self, row, item):
        return self.get_stats_func(self.acc, row, item)
    
    
class ItemCTRInSequence:
    """
    Calculates statistics of items which were clicked as the last in sequence
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.item_clicks_when_last = defaultdict(int)
        self.item_impressions_when_last = defaultdict(int)
        self.item_click_in_rev_seq = defaultdict(int)
        self.item_count_in_rev_seq = defaultdict(int)

    def update_acc(self, row):
        item_id = int(row["reference"])
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
        obs["item_average_seq_pos"] = self.item_click_in_rev_seq[item_id] / (
                    self.item_count_in_rev_seq[item_id] + 1)
        return obs
    
class ItemCTR:
    def __init__(self, action_types):
        self.action_types = action_types
        self.clicks = defaultdict(int)
        self.impressions = defaultdict(int)

    def update_acc(self, row):
        self.clicks[row["reference"]] += 1
        for item_id in row["impressions"]:
            self.impressions[item_id] += 1

    def get_stats(self, row, item):
        output = {}
        output["clickout_item_clicks"] = self.clicks[item["item_id"]]
        output["clickout_item_impressions"] = self.impressions[item["item_id"]]
        return output

feature_generator = FeatureGenerator(accumulators=[ItemCTRInSequence(), ItemCTR(action_types=["clickout item"])],
                             input="../data/events_sorted.csv", 
                             save_as="test.csv", limit=1000000)
feature_generator.generate_features()
# -

df = pd.read_csv("test.csv")

results = []
for col in tqdm(df.columns):
    if df[col].dtype in [np.int, np.float] and col != "was_clicked": # and ("rank_weighted" in col or col == "item_average_seq_pos" or "when_last" in col):
        results.append((col, mrr_fast(df, col)))
        df[col + "_rank"] = df.groupby("clickout_id")[col].rank("max", ascending=False)
        mrr_rank = mrr_fast(df, col + "_rank")
        df[col + "_rank_rev"] = df.groupby("clickout_id")[col].rank("max", ascending=True)
        mrr_rank_rev = mrr_fast(df, col + "_rank_rev")
        results.append((col + "_rank", max(mrr_rank, mrr_rank_rev)))
results_df = pd.DataFrame.from_records(results, columns=["col", "mrr"])
results_df.sort_values("mrr", ascending=False, inplace=True)
print(results_df)


results_df.to_csv("fimp.csv")

df = pd.read_csv("../data/events_sorted.csv", nrows=100000)
df["clickout_step_rev"].value_counts()

df = pd.read_csv("test.csv")
df["item_ctr_when_last"].value_counts()


