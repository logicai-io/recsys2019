from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm


def mean(xs):
    if xs:
        return sum(xs) / len(xs)
    else:
        return 0


df = pd.read_csv("../../../data/events_sorted.csv")
clickouts = df[df["action_type"] == "clickout item"]
prices_dict = defaultdict(list)

for impressions_list, prices_list in tqdm(zip(clickouts["impressions"], clickouts["prices"])):
    items_ids = list(map(int, impressions_list.split("|")))
    prices = list(map(int, prices_list.split("|")))
    for item_id, price in zip(items_ids, prices):
        prices_dict[item_id].append(price)

for k, v in prices_dict.items():
    prices_dict[k] = mean(v)

prices_dict = dict(prices_dict)

joblib.dump(prices_dict, "../../../data/item_prices.joblib", compress=3)
