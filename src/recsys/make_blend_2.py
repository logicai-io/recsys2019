import pandas as pd
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from scipy.optimize import fmin
from scipy.special import logit

logger = get_logger()

logger.info("Staring blending")

events = pd.read_csv("../../data/events_click_rev.csv")
# events = events[events["action_type"]=="clickout item"][["user_id", "session_id", "step", "clickout_step_rev"]]

def join_events(df):
    df = pd.merge(df, events, on=["user_id", "session_id", "step"])
    return df #[df["clickout_step_rev"]==1]

p_lgbr4 = join_events(pd.read_csv("predictions/model_val_7f439f9956aa97213dd15397faea4b0fef0a2152.csv"))
p_lgbr5 = join_events(pd.read_csv("predictions/model_val_7937c520a0a230a9d06b9030ebcdc73f8372305d.csv"))
p_nn = join_events(pd.read_csv("predictions/model_val_608b35325393f8133924b31272e0b4f94f828125.csv"))


final = p_lgbr4.copy()

for p in [p_lgbr4, p_lgbr5, p_nn]:
    print(mrr_fast(p, "click_proba"))

def opt(v):
    a, b, c = v
    final["click_proba"] = (
            a * p_lgbr4["click_proba"] + b * p_lgbr5["click_proba"] + c * p_nn["click_proba"]
    )
    mrr = mrr_fast(final, "click_proba")
    print(v)
    logger.info(f"MRR {mrr}")
    return -mrr


a, b, c = fmin(opt, [1, 1, 1])
final["click_proba"] = (
    a * p_lgbr4["click_proba"] + b * p_lgbr5["click_proba"] + c * p_lgbr6["click_proba"] + d * p_lgbr7["click_proba"]
)
mrr = mrr_fast(final, "click_proba")

p_lgbr4 = pd.read_csv("predictions/model_submit_b7db8d0961b87e6d5887d1363ecd8d3cd711c439.csv")
p_lgbr5 = pd.read_csv("predictions/model_submit_7937c520a0a230a9d06b9030ebcdc73f8372305d.csv")
final = p_lgbr4.copy()
final["click_proba"] = (
    a * p_lgbr4["click_proba"] + b * p_lgbr5["click_proba"]
)
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
save_as = f"submissions/submission_{mrr_str}.csv"
logger.info(f"Saving submission file {save_as}")
submission_df.to_csv(save_as, index=False)

