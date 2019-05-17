import pandas as pd
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts
from scipy.optimize import fmin

logger = get_logger()

logger.info("Staring blending")

p_lgbr = pd.read_csv("predictions/model_val_92dc89e0765dbc875a8052ae0dbdd7e5ee4271bc.csv")
p_lgbr2 = pd.read_csv("predictions/model_val_365e3484ea47c9c0e5dab466da1a441656cf13d9.csv")
p_lgbr3 = pd.read_csv("predictions/model_val_d2c1eff343a5048da32abf64f3a5daf0f7bc0025.csv")
p_lgbr4 = pd.read_csv("predictions/model_val_b7db8d0961b87e6d5887d1363ecd8d3cd711c439.csv")
print(mrr_fast(p_lgbr4, "click_proba"))
final = p_lgbr.copy()
def opt(v):
    a,b,c,d = v
    final["click_proba"] = a  * p_lgbr["click_proba"] + b * p_lgbr2["click_proba"] + c*p_lgbr3["click_proba"] + d*p_lgbr4["click_proba"]
    mrr = mrr_fast(final, "click_proba")
    print(v)
    logger.info(f"MRR {mrr}")
    return -mrr

a,b,c,d = fmin(opt, [1,1,1,1])
final["click_proba"] = a  * p_lgbr["click_proba"] + b * p_lgbr2["click_proba"] + c*p_lgbr3["click_proba"] + d*p_lgbr4["click_proba"]
mrr = mrr_fast(final, "click_proba")

p_lgbr = pd.read_csv("predictions/model_submit_92dc89e0765dbc875a8052ae0dbdd7e5ee4271bc.csv")
p_lgbr2 = pd.read_csv("predictions/model_submit_365e3484ea47c9c0e5dab466da1a441656cf13d9.csv")
p_lgbr3 = pd.read_csv("predictions/model_submit_d2c1eff343a5048da32abf64f3a5daf0f7bc0025.csv")
p_lgbr4 = pd.read_csv("predictions/model_submit_b7db8d0961b87e6d5887d1363ecd8d3cd711c439.csv")
final = p_lgbr.copy()
final["click_proba"] = a * p_lgbr["click_proba"] + b * p_lgbr2["click_proba"] + c*p_lgbr3["click_proba"] + d*p_lgbr4["click_proba"]
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
save_as = f"submissions/submission_{mrr_str}.csv"
logger.info(f"Saving submission file {save_as}")
submission_df.to_csv(save_as, index=False)
