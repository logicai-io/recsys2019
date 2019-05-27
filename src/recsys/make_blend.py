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
    return df[df["clickout_step_rev"] == 1]


p_lgbr = join_events(pd.read_csv("predictions/model_val_92dc89e0765dbc875a8052ae0dbdd7e5ee4271bc.csv"))
p_lgbr2 = join_events(pd.read_csv("predictions/model_val_365e3484ea47c9c0e5dab466da1a441656cf13d9.csv"))
p_lgbr3 = join_events(pd.read_csv("predictions/model_val_d2c1eff343a5048da32abf64f3a5daf0f7bc0025.csv"))
p_lgbr4 = join_events(pd.read_csv("predictions/model_val_b7db8d0961b87e6d5887d1363ecd8d3cd711c439.csv"))
p_lgbr5 = join_events(pd.read_csv("predictions/model_val_7937c520a0a230a9d06b9030ebcdc73f8372305d.csv"))

final = p_lgbr.copy()

for p in [p_lgbr, p_lgbr2, p_lgbr3, p_lgbr4, p_lgbr5]:
    print(mrr_fast(p, "click_proba"))


def opt(v):
    a, b, c, d, e = v
    final["click_proba"] = (
        a * p_lgbr["click_proba"]
        + b * p_lgbr2["click_proba"]
        + c * p_lgbr3["click_proba"]
        + d * p_lgbr4["click_proba"]
        + e * p_lgbr5["click_proba"]
    )
    mrr = mrr_fast(final, "click_proba")
    print(v)
    logger.info(f"MRR {mrr}")
    return -mrr


a, b, c, d, e = fmin(opt, [1, 1, 1, 1, 1])
a, b, c, d, e = fmin(opt, [a, b, c, d, e])

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
