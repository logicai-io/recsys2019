from collections import defaultdict
from csv import DictWriter

import pandas as pd
from recsys.data_generator.accumulators import ACTIONS_WITH_ITEM_REFERENCE
from tqdm import tqdm
import numpy as np
import joblib

"""
Calculates leave one out stats
"""

data = pd.read_csv("../../../data/events_sorted.csv")

stats = {}

"""
Build stats 
item_id -> stat_name -> set(users)
"""

for action_type, user_id, platform, impression, reference in tqdm(
    zip(data["action_type"], data["user_id"], data["platform"], data["impressions"], data["reference"])
):
    if reference is None or reference == np.nan or action_type != "clickout item":
        continue
    for item_id in impression.split("|"):
        item_id = int(item_id)
        try:
            stats[(item_id, platform)]["impressions"].add(user_id)
        except KeyError:
            stats[(item_id, platform)] = {"impressions": {user_id}}

for user_id, platform, reference, action_type in tqdm(
    zip(data["user_id"], data["platform"], data["reference"], data["action_type"])
):
    if reference is None or reference == np.nan:
        continue
    if action_type in ACTIONS_WITH_ITEM_REFERENCE:
        try:
            item_id = int(reference)
        except:
            continue

        if (item_id, platform) not in stats:
            stats[(item_id, platform)] = {}

        try:
            stats[(item_id, platform)][action_type].add(user_id)
        except KeyError:
            stats[(item_id, platform)][action_type] = {user_id}

joblib.dump(stats, "../../../data/item_stats_loo_by_platform.joblib")
