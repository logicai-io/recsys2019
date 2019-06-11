import pandas as pd
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast, mrr_fast_v2
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
"bdeaf88e4e4085337041afa965d69585bf8f7182",
"7ec17c9bbfae3b0d9fae0518f0cf8aa2b9069116"
]

predictions = []
for model_hash in models:
    predictions.append(join_events(pd.read_csv(f"predictions/model_val_{model_hash}.csv")))


for p in predictions:
    p.sort_values(["user_id", "session_id", "step"], inplace=True)
    p.reset_index(inplace=True, drop=True)
    print(mrr_fast(p, "click_proba"))

final = predictions[-1].copy()

def opt(v):
    final["click_proba"] = 0
    for c, pred in zip(v, predictions):
        c = max(0, c)
        final["click_proba"] += c * pred["click_proba"]
    mrr = mrr_fast_v2(final["was_clicked"], final["click_proba"], final["clickout_id"])
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
    c = max(0, c)
    final["click_proba"] += c * pred["click_proba"]
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
save_as = f"submissions/submission_{mrr_str}.csv"
logger.info(f"Saving submission file {save_as}")
submission_df.to_csv(save_as, index=False)
