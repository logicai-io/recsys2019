from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../../../data/events_sorted.csv")
clickouts = df[df["action_type"] == "clickout item"]
records = defaultdict(int)
for impressions_list, prices_list, platform in tqdm(
    zip(clickouts["impressions"], clickouts["prices"], clickouts["platform"])
):
    items_ids = list(map(int, impressions_list.split("|")))
    prices = list(map(int, prices_list.split("|")))
    for item_id, price in zip(items_ids, prices):
        records[(item_id, price)] += 1

records = [(item_id, price, freq) for (item_id, price), freq in records.items()]

df_prices = pd.DataFrame.from_records(records, columns=["item_id", "price", "freq"]).drop_duplicates()
df_prices["price_freq_rank_asc"] = df_prices.groupby("item_id")["freq"].rank("max", ascending=True)
freq_sums_by_item = dict(df_prices.groupby("item_id")["freq"].sum())
df_prices["price_freq_share"] = df_prices["freq"] / df_prices["item_id"].map(freq_sums_by_item)
df_prices["price_freq_share_rank_asc"] = df_prices.groupby("item_id")["price_freq_share"].rank("max", ascending=False)
df_prices["price_rank_asc"] = df_prices.groupby("item_id")["price"].rank("max", ascending=True)
df_prices["price_rank_desc"] = df_prices.groupby("item_id")["price"].rank("max", ascending=False)
df_prices["price_rank_asc_pct"] = df_prices.groupby("item_id")["price"].rank("max", pct=True, ascending=True)
min_price_by_hotel = df_prices.groupby("item_id")["price"].min().reset_index().rename(columns={"price": "min_price"})
max_price_by_hotel = df_prices.groupby("item_id")["price"].max().reset_index().rename(columns={"price": "max_price"})
count_price_by_hotel = (
    df_prices.groupby("item_id")["price"].count().reset_index().rename(columns={"price": "count_price"})
)
df_prices = pd.merge(df_prices, min_price_by_hotel, on="item_id")
df_prices = pd.merge(df_prices, max_price_by_hotel, on="item_id")
df_prices = pd.merge(df_prices, count_price_by_hotel, on="item_id")
df_prices["price_relative_to_min"] = df_prices["price"] / df_prices["min_price"]
df_prices["price_range"] = df_prices["max_price"] - df_prices["min_price"]
df_prices["price_range_div"] = df_prices["max_price"] / df_prices["min_price"]
df_prices.sort_values(["item_id", "price"], inplace=True)

joblib.dump(df_prices, "../../../data/item_prices_rank.joblib")