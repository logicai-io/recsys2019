import pandas as pd
from recsys.metric import mrr_fast

p_lgb = pd.read_csv("predictions/model_1_val.csv")
p_lgbr = pd.read_csv("predictions/model_2_val.csv")

final = p_lgb.copy()

final["pred"] = 1.0 * p_lgb["pred"] + 0.2 * p_lgbr["pred"]
print("MRR", mrr_fast(final, "pred"))

# p_lgb = pd.read_csv("predictions/data_py_lgbclas_test.csv")
# p_lgbr = pd.read_csv("predictions/data_py_lgbrank_test.csv")
# s_lgb = pd.read_csv("predictions/data_sc_lgbclas_test.csv")
# s_lgbr = pd.read_csv("predictions/data_sc_lgbrank_test.csv")
#
# final = p_lgb.copy()
# final["click_proba"] = 1.0 * p_lgb["pred"] + 0.2 * p_lgbr["pred"] + 0.2 * s_lgb["pred"]
#
# _, submission_df = group_clickouts(final)
# submission_df.to_csv("submission_blend.csv", index=False)
