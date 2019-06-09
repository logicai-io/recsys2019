import pandas as pd
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from scipy.optimize import fmin

logger = get_logger()

logger.info("Staring blending")

events = pd.read_csv("../../data/events_click_rev.csv")
# events = events[events["action_type"]=="clickout item"][["user_id", "session_id", "step", "clickout_step_rev"]]


def join_events(df):
    df = pd.merge(df, events, on=["user_id", "session_id", "step"])
    return df  # [df["clickout_step_rev"] == 1]


models = [
"ed093863252d7a631e4ec45585459c5c23ca556d",
"eb6c4a7ebf082f0a4e9d1de0a26563d1426b220b",
"b1ca28d7513cbfa03d789fc078bda8435491418c",
"857d156f5093ac96d6cc12c9df2064f1d7fb458d",
"7b72223b7337409b3dd08eebd4bb8b032b642bdf",
"86528c208aa207391389c0907566fc17485bb2e0"
]

predictions = []
for model_hash in models:
    predictions.append(join_events(pd.read_csv(f"predictions/model_val_{model_hash}.csv")))

final = predictions[-1].copy()

for p in predictions:
    print(mrr_fast(p, "click_proba"))


def opt(v):
    final["click_proba"] = 0
    for c, pred in zip(v, predictions):
        final["click_proba"] += c * pred["click_proba"]
    mrr = mrr_fast(final, "click_proba")
    print(v)
    logger.info(f"MRR {mrr}")
    return -mrr


coefs = fmin(opt, [0] * len(predictions))
mrr = mrr_fast(final, "click_proba")

submissions = []
for model_hash in models:
    submissions.append(join_events(pd.read_csv(f"predictions/model_submit_{model_hash}.csv")))
final = submissions[-1].copy()
final["click_proba"] = 0
for c, pred in zip(coefs, submissions):
    final["click_proba"] += c * pred["click_proba"]
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
save_as = f"submissions/submission_{mrr_str}.csv"
logger.info(f"Saving submission file {save_as}")
submission_df.to_csv(save_as, index=False)


"""
final["click_proba"] = (
    a * p_lgbr["click_proba"]
    + b * p_lgbr2["click_proba"]
    + c * p_lgbr3["click_proba"]
    + d * p_lgbr4["click_proba"]
    + e * p_lgbr5["click_proba"]
)
mrr = mrr_fast(final, "click_proba")

p_lgbr = pd.read_csv("predictions/model_submit_92dc89e0765dbc875a8052ae0dbdd7e5ee4271bc.csv")
p_lgbr2 = pd.read_csv("predictions/model_submit_365e3484ea47c9c0e5dab466da1a441656cf13d9.csv")
p_lgbr3 = pd.read_csv("predictions/model_submit_d2c1eff343a5048da32abf64f3a5daf0f7bc0025.csv")
p_lgbr4 = pd.read_csv("predictions/model_submit_b7db8d0961b87e6d5887d1363ecd8d3cd711c439.csv")
p_lgbr5 = pd.read_csv("predictions/model_submit_7937c520a0a230a9d06b9030ebcdc73f8372305d.csv")
final = p_lgbr.copy()
final["click_proba"] = (
    a * p_lgbr["click_proba"]
    + b * p_lgbr2["click_proba"]
    + c * p_lgbr3["click_proba"]
    + d * p_lgbr4["click_proba"]
    + e * p_lgbr5["click_proba"]
)
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
save_as = f"submissions/submission_{mrr_str}.csv"
logger.info(f"Saving submission file {save_as}")
submission_df.to_csv(save_as, index=False)
"""
