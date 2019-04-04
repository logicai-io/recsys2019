import pandas as pd
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts

p_lgbr = pd.read_csv("predictions/model_2_val.csv")
final = p_lgbr.copy()
final["click_proba"] = 1.0 * p_lgbr["click_proba"]
mrr = mrr_fast(final, "click_proba")
print("MRR", mrr)

p_lgbr = pd.read_csv("predictions/model_2_submit.csv")
final = p_lgbr.copy()
final["click_proba"] = 1.0 * p_lgbr["click_proba"]
_, submission_df = group_clickouts(final)
mrr_str = str(mrr).split(".")[1][:4]
submission_df.to_csv(f"submissions/submission_{mrr_str}.csv", index=False)
