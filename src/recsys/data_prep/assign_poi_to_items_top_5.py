from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../../../data/events_sorted.csv")
pois = df[df["action_type"] == "search for poi"]
pois = pois[~pois["fake_impressions"].isnull()]
pois["fake_impressions"] = pois["fake_impressions"].map(lambda x: x.split("|"))

item_pois = defaultdict(set)

for i, row in tqdm(pois.iterrows()):
    for item_id in row["fake_impressions"][:5]:
        item_pois[int(item_id)].add(row["reference"])

item_pois_text = defaultdict(str)

for item_id, poi_set in tqdm(item_pois.items()):
    item_pois_text[item_id] = " ".join(list(poi_set))

joblib.dump(item_pois, "../../../data/item_pois_text_top_5.joblib", compress=3)
