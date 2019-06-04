from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm

metadata_dense = pd.read_csv("../../../data/item_metadata_dense.csv")
item_rating = dict(zip(metadata_dense["item_id"], metadata_dense["rating"].fillna(0.0)))
item_rating = defaultdict(float, item_rating)


def sort_by_rating(item_ids):
    return sorted(item_ids, key=lambda x: item_rating[x], reverse=True)


df = pd.read_csv("../../../data/events_sorted.csv")

for criterium in ["Sort By Rating", "Sort By Distance", "Sort By Popularity"]:
    print(criterium)
    name = criterium.lower().replace(" ", "_")

    df_sort_by_rating = df[(df["current_filters"].str.find(criterium) >= 0) & (df["action_type"] == "clickout item")]
    df_sort_by_rating["impressions_parsed"] = (
        df_sort_by_rating["impressions"].str.split("|").map(lambda x: list(map(int, x)))
    )

    ordered_pairs = set()
    all_items = set()
    for impressions in tqdm(df_sort_by_rating["impressions_parsed"]):
        for idx_a, item_a in enumerate(impressions):
            for idx_b, item_b in enumerate(impressions[(idx_a + 1) :]):
                ordered_pairs.add((item_a, item_b))
                all_items.add(item_a)
                all_items.add(item_b)

    good_item = defaultdict(int)
    bad_item = defaultdict(int)

    for item_a, item_b in ordered_pairs:
        good_item[item_a] += 1
        bad_item[item_b] += 1

    def calc_rating(item_id):
        return good_item[item_id] / (good_item[item_id] + bad_item[item_id])

    stats = defaultdict(float)
    for item_id in all_items:
        stats[item_id] = calc_rating(item_id)

    joblib.dump(stats, f"../../../data/item_{name}_stats.joblib")
