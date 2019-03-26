from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../../../data/item_metadata.csv")

features_map = {}
hotels = defaultdict(set)
for i, row in tqdm(df.iterrows()):
    for feature in row["properties"].split("|"):
        if feature not in features_map:
            features_map[feature] = len(features_map)
        hotels[row["item_id"]].add(features_map[feature])

joblib.dump(hotels, "../../../data/item_metadata_map.joblib", compress=3)
