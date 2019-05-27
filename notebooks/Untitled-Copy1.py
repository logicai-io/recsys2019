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
from collections import defaultdict
from recsys.metric import mrr_fast_v2, mrr_fast
import numpy as np


# +
class CTRTrigrams:
    def __init__(self):
        self.action_types = ["clickout item"]
        self.trigram_clicks = defaultdict(int)
        self.trigram_set_clicks = defaultdict(lambda: defaultdict(int))
        self.trigram_set_impressions = defaultdict(int)
        
    def update_acc(self, row):
        if row["index_clicked"] == -1000:
            return
        impressions = ["FIRST"] + row["impressions"] + ["END"]
        index = row["index_clicked"] + 1
        trigram_clicked = (impressions[index-1], impressions[index], impressions[index+1])
        self.trigram_clicks[trigram_clicked] += 1
        trigram_clicked_set = tuple(sorted(trigram_clicked))
        self.trigram_set_clicks[trigram_clicked_set][row["reference"]] += 1
    
        for n in range(1, len(impressions)-1):
            trigram_impression = (impressions[n-1], impressions[n], impressions[n+1])
            trigram_impression_set = tuple(sorted(trigram_impression))
            self.trigram_set_impressions[trigram_impression_set] += 1
    
    def get_stats(self, row, item):
        impressions = ["FIRST"] + row["impressions"] + ["END"]
        index = item["rank"]
        trigram_clicked = (impressions[index-1], impressions[index], impressions[index+1])
        trigram_set_clicked = tuple(sorted(trigram_clicked))
        output = {}
        output["trigram_clicks"] = self.trigram_clicks[trigram_clicked]
        output["trigram_set_clicks"] = self.trigram_set_clicks[trigram_set_clicked][item["item_id"]]
        output["trigram_set_ctr"] = self.trigram_set_clicks[trigram_set_clicked][item["item_id"]] / (1+self.trigram_set_impressions[trigram_set_clicked])
        return output
    
class GlobalTimestampPerItem:
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
    
    
def mean(xs):
    if len(xs) > 0:
        return sum(xs) / len(xs)
    else:
        return 0
    
class AverageItemRanking:
    def __init__(self):
        self.action_types = ["clickout item"]
        self.rankings = defaultdict(list)
        
    def update_acc(self, row):
        for rank, item in enumerate(row["impressions"]):
            self.rankings[item].append(rank)
        
    def get_stats(self, row, item):
        output = {}
        if self.rankings[item["item_id"]]:
            output["average_ranking_alltime"] = mean(self.rankings[item["item_id"]])
            output["average_ranking_last_10"] = mean(self.rankings[item["item_id"]][:10])
            output["average_ranking_last_100"] = mean(self.rankings[item["item_id"]][:100])
        else:
            output["average_ranking_alltime"] = 12.5
            output["average_ranking_last_10"] = 12.5
            output["average_ranking_last_100"] = 12.5            
        return output
    


class SameImpressionUserStats:
    """
    Example definition

    StatsAcc(filter=lambda row: row.action_type == "clickout_item",
             init_acc=defaultdict(int),
             updater=lambda acc, row: acc[(row.user_id, row.item_id)]+=1)
    """

    def __init__(self):
        self.action_types = ["clickout item"]
        self.view_stats = defaultdict(lambda: defaultdict(set))

    def update_acc(self, row):
        self.view_stats[row["impressions_raw"]][row["user_id"]]
        self.updater(self.acc, row)

    def get_stats(self, row, item):
        return self.get_stats_func(self.acc, row, item)

    
feature_generator = FeatureGenerator(accumulators=[AverageItemRanking()],
                             input="../data/events_sorted.csv", 
                             save_as="test.csv", limit=1000000)
feature_generator.generate_features()
# -

df = pd.read_csv("test.csv")
df["average_ranking_alltime_neg"] = -df["average_ranking_alltime"]

print(mrr_fast(df.query("average_ranking_alltime != 12.5"), "average_ranking_alltime"))
print(mrr_fast(df.query("average_ranking_alltime != 12.5"), "average_ranking_alltime_neg"))

mrr_fast_v2(df["was_clicked"], -df["average_ranking_alltime"], df["clickout_id"])

print(mrr_fast(df, "last_item_time_diff"))

mrr_fast(df.query("last_item_time_diff_same_user < 100"), "last_item_time_diff_same_user")

df.query("last_item_time_diff_same_user < 10")[["reference", "last_item_last_user_id", "last_item_time_diff", "user_id", "rank", "was_clicked"]]

df[["rank", "average_ranking"]]


