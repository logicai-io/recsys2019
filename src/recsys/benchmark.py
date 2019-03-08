from collections import defaultdict

import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../../data/train_sample_5.csv")
df.sort_values(["timestamp", "user_id", "step"], inplace=True)

item_impressions_counter = defaultdict(int)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if row["action_type"] == "clickout item":
        impressions = row["impressions"].split("|")
        for item_id in impressions:
            item_impressions_counter[item_id] += 1
