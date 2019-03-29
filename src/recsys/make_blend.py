import pandas as pd
from recsys.metric import mrr_fast
from recsys.submission import group_clickouts

p_lgb = pd.read_csv("predictions/model_1_val.csv")
p_lgbr = pd.read_csv("predictions/model_2_val.csv")

final = p_lgb.copy()

final["click_proba"] = 0.0 * p_lgb["click_proba"] + 1.0 * p_lgbr["click_proba"]
print("MRR", mrr_fast(final, "click_proba"))

p_lgb = pd.read_csv("predictions/model_2_submit.csv")
p_lgbr = pd.read_csv("predictions/model_2_submit.csv")
# s_lgb = pd.read_csv("predictions/data_sc_lgbclas_test.csv")
# s_lgbr = pd.read_csv("predictions/data_sc_lgbrank_test.csv")

final = p_lgb.copy()
final["click_proba"] = 0.0 * p_lgb["click_proba"] + 1.0 * p_lgbr["click_proba"]
_, submission_df = group_clickouts(final)
submission_df.to_csv("submission_blend.csv", index=False)
