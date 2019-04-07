import pandas as pd
from recsys.log_utils import get_logger
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts

logger = get_logger()

logger.info("Staring blending")

p_lgbr = pd.read_csv("predictions/model_2_val.csv")
final = p_lgbr.copy()
final["click_proba"] = 1.0 * p_lgbr["click_proba"]
mrr = mrr_fast(final, "click_proba")
logger.info("MRR", mrr)

p_lgbr = pd.read_csv("predictions/model_2_submit.csv")
final = p_lgbr.copy()
final["click_proba"] = 1.0 * p_lgbr["click_proba"]
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
save_as = f"submissions/submission_{mrr_str}.csv"
logger.info(f"Saving submission file {save_as}")
submission_df.to_csv(save_as, index=False)
